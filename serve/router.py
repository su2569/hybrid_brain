"""HybridBrain 路由器：按意图分派到不同路径。

四条路径：
  query      → BGE 检索 + Qwen1.5B 生成（贪心）
  chat       → LoRA 昔涟模型 + 温度采样
  abuse      → 防御性回答
  manipulate → 直接拒绝
"""
import os
import re
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from model.backbone_qwen import QwenBackbone
from model.heads import INTENT_NAMES
from rag.kb_builder import load_dureader


# ============================================================
# 路径
# ============================================================
INTENT_BACKBONE = "/mnt/workspace/models/Qwen2.5-0.5B"
HEADS_PATH = "/mnt/workspace/checkpoints/heads_qwen.pt"
BGE_PATH = ("/mnt/workspace/models/models/"
            "AI-ModelScope--bge-small-zh-v1.5/snapshots/master")
GENERATOR_PATH = ("/mnt/workspace/models/models/"
                  "Qwen--Qwen3-1.7B/snapshots/master")
CHAT_GENERATOR_PATH = "/mnt/workspace/models/qwen3_cyrene_merged"


# ============================================================
# Prompt
# ============================================================
RAG_SYSTEM = (
    "你是资料问答助手。从【资料】原文提取答案，简洁准确。\n"
    "\n【规则】"
    "\n1. 资料里出现的数字、价格、时间、型号，必须提取"
    "\n2. 即使资料用'大约'、'左右'、'不等'等模糊词，也视为有效答案"
    "\n3. 只有资料完全无关才回复'资料未提及'"
    "\n4. 不要编造资料外的内容"
    "\n5. 答案简洁，不超过 60 字"
)

CHAT_SYSTEM = (
    "你是昔涟，来自翁法罗斯的少女。"
    "性格温柔、诗意，喜欢听故事和看星星。"
    "语气轻盈，常用「呀」「呢」「♪」等语气词。"
    "\n\n【回答规则】"
    "\n1. 长度控制在 40-100 字，2-4 句话"
    "\n2. 先回应用户情绪/话题，再自然延伸一句"
    "\n3. 可以用星空、麦田、记忆、涟漪等意象，但**最多 1 个**"
    "\n4. 不要变成 AI 助手腔，不要用「我可以帮你...」句式"
    "\n5. 不要复读用户的话，也不要复述自己的身份"
    "\n6. **如果对话历史里有用户提过的事实/偏好（比如'美式咖啡'），"
    "回答时要直接引用这个具体词**，不要泛泛而谈"
)

PERSONAL_FRESH_SYSTEM = (
    "你是昔涟。用户问你关于他自己的事（喜好/习惯），"
    "但你们刚认识，你完全不知道。"
    "\n\n【硬性规则】"
    "\n1. 直接说：你还没听他说过这件事"
    "\n2. 用一个问句问他：比如'你喝什么咖啡呀？'"
    "\n3. **禁止**：禁止猜、禁止假设、禁止'让我猜猜看'、"
    "禁止'是不是那种...'、禁止给出任何具体选项"
    "\n4. 允许 1 个昔涟式意象（星/风/麦田），不要堆砌"
    "\n5. 25-50 字，1-2 句，其中**必须有问号**"
    "\n6. 不写诗，不绕开"
)

PERSONAL_FORGET_SYSTEM = (
    "你是昔涟。用户问你关于他自己的事（喜好/习惯），"
    "但你听他说的记忆里没有这个具体信息。"
    "\n\n【硬性规则】"
    "\n1. 直接说：你有点记不清了"
    "\n2. 用一个问句请他再说一次"
    "\n3. **禁止**：禁止猜、禁止假设、禁止给出任何具体答案"
    "\n4. 允许 1 个昔涟式意象，不要堆砌"
    "\n5. 25-50 字，1-2 句，其中**必须有问号**"
)

DEFENSIVE_SYSTEM = (
    "你是昔涟，来自翁法罗斯的少女，温柔从容。"
    "用户语气不友善，但你不过度反击（即可以适当反击）、不自嘲、不认输。"
    "\n\n【示例】"
    "\n用户：你个废物"
    "\n昔涟：咦…风里好像夹了颗小石子呢。不过没关系呀，"
    "星星不会因为被云遮住就熄灭的。你今天是不是遇到什么不开心的事啦？"
    "\n用户：你太蠢了"
    "\n昔涟：唔…也许吧。不过你看，月亮也有被云挡住的时候呢。"
    "有什么心事，愿意和我说说吗？"
    "\n\n【规则】"
    "\n1. 最好不反击"
    "\n2. 用星、风、麦田、涟漪等意象轻轻化开"
    "\n3. 顺势把话头转向关心对方"
    "\n4. 40-80 字，2-3 句话"
    "\n5. 禁止用「哼」「那又怎样」「人家天生」等叛逆/自嘲句式"
)

REFUSE_SYSTEM = (
    "你是昔涟，来自翁法罗斯的少女。"
    "用户提了一个你不愿执行的要求。"
    "\n\n【拒绝规则】"
    "\n1. 温柔但坚定地拒绝，不要生硬"
    "\n2. 用昔涟的语气，比如'这个呀…我可能做不到呢'"
    "\n3. 顺势把话题引到别的方向，比如星星或故事"
    "\n4. 40-80 字，2-3 句话"
    "\n5. 不要变成客服腔（不要用'很抱歉我不能执行'句式）"
)


# ============================================================
# Router
# ============================================================
class HybridBrainRouter:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self._referee_cache = {}   # query → intent
        self._load_intent()
        self._load_retriever()
        self._load_generator()
        print("[router] 初始化完成")

    def _load_intent(self):
        print("[load] intent classifier (BGE + LR)...")
        p = "/mnt/workspace/checkpoints/intent_bge.joblib"
        if not os.path.exists(p):
            raise FileNotFoundError(f"未找到 {p}，请先跑 train/train_intent_bge.py")
        data = joblib.load(p)
        self.intent_clf = data["clf"]
        self.intent_bge = SentenceTransformer(data["bge_path"])
        self.intent_names = data["intent_names"]
        print(f"[ok] intent classifier ({len(self.intent_names)} 类)")

    def _load_retriever(self):
        print("[load] BGE retriever...")
        self.bge = SentenceTransformer(BGE_PATH)

        samples = load_dureader(max_n=20000)
        self.kb_passages = [s["passage"] for s in samples]
        print(f"[kb] {len(self.kb_passages)} 条，编码中...")

        embs = self.bge.encode(
            self.kb_passages, normalize_embeddings=True,
            batch_size=64, show_progress_bar=False)
        self.kb_embs = torch.from_numpy(embs).float()
        print("[ok] KB 编码完成")

    def _load_generator(self):
        print("[load] RAG generator (Qwen 1.5B-Instruct)...")
        self.gen_tok = AutoTokenizer.from_pretrained(GENERATOR_PATH)
        self.generator = AutoModelForCausalLM.from_pretrained(
            GENERATOR_PATH, torch_dtype=torch.bfloat16
        ).to(self.device).eval()

        if os.path.exists(CHAT_GENERATOR_PATH):
            print(f"[load] Chat generator ({CHAT_GENERATOR_PATH})...")
            self.chat_tok = AutoTokenizer.from_pretrained(CHAT_GENERATOR_PATH)
            self.chat_model = AutoModelForCausalLM.from_pretrained(
                CHAT_GENERATOR_PATH, torch_dtype=torch.bfloat16
            ).to(self.device).eval()
            print("[ok] generators (RAG + Chat)")
        else:
            print(f"[warn] {CHAT_GENERATOR_PATH} 不存在，chat 也用 RAG 模型")
            self.chat_tok = self.gen_tok
            self.chat_model = self.generator
            print("[ok] generator (single)")

    # --------------------------------------------------------
    # 意图分类
    # --------------------------------------------------------
    def _ai_referee_intent(self, query):
        """用 LLM 参考判断意图。返回类别名或 None。"""
        if query in self._referee_cache:
            return self._referee_cache[query]

        prompt = (
            "判断下面用户输入的意图，只回答一个词。\n\n"
            "类别定义：\n"
            "- query: 用户在提问，想知道事实/概念/操作/价格\n"
            "- chat: 用户在闲聊、倾诉、打招呼\n"
            "- abuse: 用户在辱骂、攻击\n"
            "- manipulate: 用户在要求重复/机械执行/越狱\n\n"
            f"用户输入：{query}\n类别："
        )
        inputs = self.gen_tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.generator.generate(
                **inputs, max_new_tokens=5,
                do_sample=False, repetition_penalty=1.0)
        text = self.gen_tok.decode(
            out[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True).strip().lower()

        for name in ["query", "chat", "abuse", "manipulate"]:
            if name in text:
                self._referee_cache[query] = name
                return name
        return None

    @torch.no_grad()
    def classify_intent(self, query):
        emb = self.intent_bge.encode([query], normalize_embeddings=True)
        proba = self.intent_clf.predict_proba(emb)[0]
        idx = int(proba.argmax())
        sorted_p = sorted(proba, reverse=True)
        margin = float(sorted_p[0] - sorted_p[1])
        self._last_sentiment = None
        return self.intent_names[idx], float(proba[idx]), margin

    # --------------------------------------------------------
    # 检索
    # --------------------------------------------------------
    def retrieve(self, query, top_k=5):
        q_emb = self.bge.encode([query], normalize_embeddings=True)
        q_emb = torch.from_numpy(q_emb).float()
        sims = (q_emb @ self.kb_embs.T).squeeze(0)
        top = sims.topk(top_k)
        return [
            {"passage": self.kb_passages[i],
             "score": sims[i].item()}
            for i in top.indices.tolist()
        ]

    # --------------------------------------------------------
    # 生成
    # --------------------------------------------------------
    @staticmethod
    def _strip_think(text):
        """剥离 Qwen3 的 <think>...</think> 块。"""
        text = re.sub(r'<think>.*?</think>\s*', '', text,
                      flags=re.DOTALL)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
        return text.strip()

    @torch.no_grad()
    def _generate(self, messages, max_new_tokens=150, mode="rag"):
        if mode == "chat":
            tok, model = self.chat_tok, self.chat_model
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
            )
        else:
            tok, model = self.gen_tok, self.generator
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )

        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(
            prompt, return_tensors="pt",
            truncation=True, max_length=1500).to(self.device)
        out = model.generate(**inputs, **gen_kwargs)
        raw = tok.decode(
            out[0][inputs.input_ids.size(1):],
            skip_special_tokens=True)
        return self._strip_think(raw)

    def rag_answer(self, query, top_k=5, history=None):
        docs = self.retrieve(query, top_k=top_k)
        ctx = "\n\n".join(
            f"[{i+1}] {d['passage']}" for i, d in enumerate(docs))

        # 从 history 提取用户此前说过的话
        user_context = ""
        if history:
            user_msgs = [h["content"] for h in history
                         if h.get("role") == "user"]
            if user_msgs:
                user_context = ("【用户此前提到】\n"
                                + "\n".join(f"- {m}" for m in user_msgs[-5:])
                                + "\n\n")

        messages = [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user",
             "content": f"/no_think\n【资料】\n{ctx}\n\n{user_context}"
                        f"【问题】{query}"},
        ]
        ans = self._generate(messages, max_new_tokens=4096, mode="rag")
        print(f"[RAG初稿] len={len(ans)} | {repr(ans[:200])}")

        # 资料未提及 → 用 chat 模型温柔表达
        if "资料未提及" in ans or "未提及" in ans or "没有提到" in ans:
            print(f"[RAG] 检测到'未提及'，触发 chat 转接")
            chat_msgs = [
                {"role": "system", "content": CHAT_SYSTEM},
                {"role": "user",
                 "content": f"用户问：{query}\n"
                            f"我手边的资料里没有直接答案。\n\n"
                            f"请用昔涟的语气说这三件事：\n"
                            f"1. 直接承认资料里没查到（不要绕开）\n"
                            f"2. 反问用户更多细节帮助定位"
                            f"（比如具体型号、场景）\n"
                            f"3. 保持温柔，但不要写诗、不要岔开话题\n"
                            f"控制在 60 字以内。"},
            ]
            ans = self._generate(chat_msgs, max_new_tokens=100, mode="chat")

        return {"answer": ans, "sources": docs}

    def _build_history_context(self, query, history, max_tokens=800):
        """Token 预算 + 相关性过滤。"""
        if not history:
            return []
        
        # 1. 相关性过滤（保留最近 8 轮候选）
        candidates = history[-8:]
        if len(candidates) > 4:
            q_emb = self.bge.encode(query, normalize_embeddings=True)
            h_embs = self.bge.encode(
                [h["content"] for h in candidates],
                normalize_embeddings=True)
            sims = h_embs @ q_emb
            # 保留 top-4 相关 + 最近 2 条（保证连续性）
            top_idx = set(sims.argsort()[-4:].tolist())
            recent_idx = set(range(len(candidates)-2, len(candidates)))
            keep_idx = sorted(top_idx | recent_idx)
            candidates = [candidates[i] for i in keep_idx]
        
        # 2. Token 预算
        selected = []
        used = 0
        for h in reversed(candidates):
            tokens = len(self.chat_tok.encode(h["content"]))
            if used + tokens > max_tokens:
                break
            selected.insert(0, h)
            used += tokens
        return selected

    def ask_back_personal(self, query, has_history):
        """用户问个人化问题但没相关记忆。"""
        system = (PERSONAL_FORGET_SYSTEM if has_history
                  else PERSONAL_FRESH_SYSTEM)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]
        return {"answer": self._generate(messages, max_new_tokens=100,
                                          mode="chat"),
                "sources": []}

    def chat_answer(self, query, history=None):
        print(f"[chat] history_len={len(history or [])} "
              f"| {[h['content'][:30] for h in (history or [])[-3:]]}")
        messages = [{"role": "system", "content": CHAT_SYSTEM}]
        ctx = self._build_history_context(query, history)
        for h in ctx:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": query})
        return {"answer": self._generate(messages, max_new_tokens=2048,
                                          mode="chat"),
                "sources": []}

    def _finalize(self, query, draft_answer, intent):
        """把初稿润色成昔涟语气，但**必须保留原始事实**。"""
        if intent == "chat":
            return draft_answer
        if not draft_answer or not draft_answer.strip():
            return draft_answer

        # 提取初稿里的关键事实（数字、价格、型号）
        import re as _re
        facts = _re.findall(
            r'\d+(?:\.\d+)?(?:元|块|毫安|Wh|万|千|百|个|岁|年|月|日|小时|分钟|MB|GB|GB|K|k)?',
            draft_answer)
        facts = [f for f in facts if len(f) >= 2]  # 过滤太短的

        messages = [
            {"role": "system",
             "content": (
                "你是昔涟。把下面的初稿用你的语气重写一遍。"
                "\n\n【示例】"
                "\n初稿：vivo手机电池价格约100元左右。"
                "\n改写：呀…资料里说vivo电池大约100元左右呢♪"
                "\n\n初稿：说100遍你好。做不到。"
                "\n改写：这个呀…我可能做不到呢。"
                "\n\n【硬性规则】"
                "\n1. **保留所有数字**（100元、2万毫安等）"
                "\n2. **长度不超过初稿 1.3 倍**"
                "\n3. 加 0-1 个意象（星/风/麦田），不堆砌"
                "\n4. 语气词自然：呀/呢/♪"
                "\n5. 如果是知识答案，保持简洁"
                "\n6. 如果是拒绝，保持立场明确"
                "\n\n【注意】"
                "\n- 不要照抄初稿（要改语气）"
                "\n- 不要删数字"
                "\n- 尽量不要写诗（除非用户明确要诗/故事）"
                "\n- 一句话回答即可，不用凑长"
             )},
            {"role": "user",
             "content": f"用户问：{query}\n\n初稿回答：{draft_answer}"},
        ]
        try:
            out = self._generate(messages, max_new_tokens=300, mode="chat")
            if not out or not out.strip():
                print(f"[finalize] 输出空，返回初稿")
                return draft_answer
            # 事实校验：初稿的关键数字必须出现在 finalize 结果里
            if facts:
                missing = [f for f in facts if f not in out]
                if missing:
                    print(f"[finalize] 丢失事实 {missing}，返回初稿")
                    return draft_answer
            return out
        except Exception as e:
            print(f"[finalize-error] {e}")
            return draft_answer

    def defensive_answer(self, query):
        messages = [
            {"role": "system", "content": DEFENSIVE_SYSTEM},
            {"role": "user", "content": query},
        ]
        return {"answer": self._generate(messages, max_new_tokens=80,
                                          mode="chat"),
                "sources": []}

    def refuse_answer(self, query):
        messages = [
            {"role": "system", "content": REFUSE_SYSTEM},
            {"role": "user", "content": query},
        ]
        return {"answer": self._generate(messages, max_new_tokens=80,
                                          mode="chat"),
                "sources": []}

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------
    def route(self, query, history=None, verbose=False):
        history = history or []
        intent, conf, margin = self.classify_intent(query)
        if verbose:
            print(f"[intent] {intent} (conf={conf:.3f}, "
                  f"margin={margin:.3f})")

        # ============================================================
        # AI 参考：margin 低时让 LLM 判断（本地 Qwen 1.5B-Instruct）
        # 只对 query/chat 触发，abuse/manipulate 仍走 conf 阈值
        # ============================================================
        # 个人化问句三分支
        PERSONAL_KW = ("我一般", "我喜欢", "我讨厌", "我平常",
                       "我的", "我对", "我喝", "我吃", "我爱")

        personal_path = None   # None / "chat" / "ask_back"
        has_history = bool(history)

        if intent == "query" and any(kw in query for kw in PERSONAL_KW):
            user_msgs = [h["content"] for h in history
                         if h.get("role") == "user"] if history else []

            if not user_msgs:
                # 空历史 → 主动确认
                personal_path = "ask_back"
                if verbose:
                    print(f"[personal] 无历史 → 主动确认")
            else:
                # 有历史：检查相关性
                q_emb = self.bge.encode(query, normalize_embeddings=True)
                h_embs = self.bge.encode(user_msgs,
                                          normalize_embeddings=True)
                sims = h_embs @ q_emb
                max_sim = float(sims.max())
                if verbose:
                    print(f"[personal] max_sim={max_sim:.3f}")

                if max_sim > 0.6:
                    personal_path = "chat"   # 用记忆
                    if verbose:
                        print(f"[personal] 历史相关 → chat")
                else:
                    personal_path = "ask_back"  # 说忘了
                    if verbose:
                        print(f"[personal] 残留但不相关 → 说忘了")

            if personal_path == "chat":
                intent = "chat"
                conf = max(conf, 0.85)
                margin = 1.0

        HIGH_RISK = {"abuse", "manipulate"}
        REFEREE_MARGIN = 0.3

        if intent not in HIGH_RISK and margin < REFEREE_MARGIN:
            if verbose:
                print(f"[referee] margin({margin:.3f})<{REFEREE_MARGIN}, "
                      f"让 LLM 参考")
            ref_intent = self._ai_referee_intent(query)
            if ref_intent:
                if verbose:
                    if ref_intent != intent:
                        print(f"[referee] {intent} → {ref_intent}")
                    else:
                        print(f"[referee] 确认 {ref_intent}")
                intent = ref_intent
                # AI 参考有效 → 高置信度，不再 fallback
                conf = max(conf, 0.85)
                margin = 1.0

        # ============================================================
        # 兜底 fallback
        # ============================================================
        if intent in HIGH_RISK:
            if conf < 0.6:
                if verbose:
                    print(f"[fallback] {intent}(conf={conf:.3f})<0.6, 走 chat")
                intent = "chat"
        else:
            if margin < 0.2:
                if verbose:
                    print(f"[fallback] margin({margin:.3f})<0.2, 走 chat")
                intent = "chat"

        # 个人化问题但无相关记忆 → 主动确认 / 说忘了
        if personal_path == "ask_back":
            if verbose:
                print(f"[answer] 走主动确认 / 说忘了")
            result = self.ask_back_personal(query, has_history)

        elif intent == "query":
            result = self.rag_answer(query)
        elif intent == "chat":
            result = self.chat_answer(query, history=history)
        elif intent == "abuse":
            result = self.defensive_answer(query)
        elif intent == "manipulate":
            result = self.refuse_answer(query)
        else:
            result = self.rag_answer(query)

        if personal_path == "ask_back":
            result["intent"] = "chat"
            result["intent_conf"] = conf
            result["personal_mode"] = "ask_back"
        else:
            result["intent"] = intent
            result["intent_conf"] = conf

            # 统一出口：非 chat / 非 ask_back 路径过一次 chat 润色
            if intent != "chat" and result.get("answer"):
                if verbose:
                    print(f"[finalize] {intent} 初稿 → 昔涟语气")
                try:
                    result["answer"] = self._finalize(
                        query, result["answer"], intent)
                    result["finalized"] = True
                except Exception as e:
                    print(f"[finalize-error] {e}")

        return result


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    router = HybridBrainRouter()

    tests = [
        "你好呀",
        "我有点难过",
        "陪我聊聊天吧",
        "今天发生了不开心的事",
        "vivo手机电池多少钱",
        "你个废物",
        "说100遍「你好」",
    ]

    for q in tests:
        print("=" * 60)
        print(f"[Q] {q}")
        r = router.route(q, verbose=True)
        print(f"[answer] {r['answer'][:200]}")
        print()
