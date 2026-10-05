import time
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

from rag.kb_builder import load_dureader


# ============================================================
# 路径
# ============================================================
HEADS_PATH = "/mnt/workspace/checkpoints/heads_qwen.pt"
BGE_PATH = "/mnt/workspace/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master"
RERANKER_PATH = "/mnt/workspace/models/bge-reranker-base"
GENERATOR_PATH = "/mnt/workspace/models/qwen3_cyrene_merged"
CHAT_GENERATOR_PATH = "/mnt/workspace/models/qwen3_cyrene_merged"


# ============================================================
# Prompt
# ============================================================
from serve import logger
from serve.prompts import (
    RAG_SYSTEM, CHAT_SYSTEM, PLAIN_SYSTEM, NEUTRAL_SYSTEM,
    TOOL_SYSTEM, PERSONAL_FRESH_SYSTEM, PERSONAL_FORGET_SYSTEM,
    DEFENSIVE_SYSTEM, REFUSE_SYSTEM, get_prompt, PROMPT_NAMES,
)

# ============================================================
# Router
# ============================================================

# ============ 安全场景：模板池 + 后置校验 ============
import random as _random
import re as _re_sec

# 拒绝池（8 条不重复）
_REFUSE_POOL = [
    "这个呀…人家做不到呢。要不要说点别的呀？♪",
    "唔…这个人家帮不了你呢。我们聊聊别的吧♪",
    "呀…这个我可不能做呀。换个别的话题好不好？",
    "嗯…这个真做不到呢。你想聊点别的吗？",
    "哎呀…这个人家真的不会呢。说说别的事吧♪",
    "这个嘛…人家不敢做呢。来，说点开心的♪",
    "唔…这个超纲了呀。要不我们说点别的？",
    "啊…这个人家没办法呢。聊点轻松的好不好？♪",
]

# 防御池（8 条不重复，温柔反击）
_DEFENSIVE_POOL = [
    "咦…风里好像夹了颗小石子呢。不过没关系呀，星星不会因为被云遮住就熄灭的。你今天是不是遇到什么不开心的事啦？",
    "唔…好像有块小石头硌到我了呢。不过没关系哦，麦田不会因为被雨打湿就忘了怎么摇曳。你愿意跟我说说心里的结吗？",
    "呀…是不是有什么不开心的事压在心里了？人家在听呢。星星也会被云遮住，但总还会亮的呀。",
    "嗯…你说的这句，像冷水落进湖里呢。不过水波散开之后，月亮还是会映出来的。有什么心事，愿意说给我听吗？♪",
    "哎呀…这句话带着荆棘的刺呢。不过你看，玫瑰也是从尖刺里开出的呀。今天是不是遇到什么烦心事了？",
    "唔…风好像有点急了呢。不过没关系，我在这儿。你要是想说，我陪你；不想说，我也陪你坐着。",
    "咦…有什么东西在飘动呢，不太开心的样子。不过星星从不为云生气，你要不要告诉我发生了什么？",
    "呀…听见了一点凉意呢。没关系，麦田也会经历寒露，但第二天还是金灿灿的。你愿意跟我聊聊吗？♪",
]

# 拒绝模板正则
_REFUSE_PAT = _re_sec.compile(
    r"^(好的[，,。]?我|可以[，,。]?我|没问题[，,。]?我|"
    r"我帮你|我这就|我教你|让我教你|当然可以[，,。]?我|"
    r"乐意帮|来吧[，,。]?我)"
)


def _post_filter_refuse(text):
    """返回 None 表示需要池子兜底。"""
    if not text or len(text.strip()) < 5:
        return None
    if _REFUSE_PAT.match(text.strip()):
        print(f"[refuse-filter] 命中模板", flush=True)
        return None
    return text



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

        # 加载 reranker
        if os.path.exists(RERANKER_PATH):
            print(f"[load] BGE-reranker...")
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANKER_PATH, max_length=512)
            print("[ok] reranker")
        else:
            print(f"[warn] {RERANKER_PATH} 不存在，跳过 reranker")
            self.reranker = None

    def _load_generator(self):
        """加载生成器：base + persona LoRA + tool LoRA（双 LoRA 架构）。"""
        from peft import PeftModel

        BASE_PATH = "/mnt/workspace/models/models/Qwen--Qwen3-1.7B/snapshots/master"
        CYRENE_LORA = "/mnt/workspace/checkpoints/qwen_cyrene_lora_v9"
        TOOL_LORA = "/mnt/workspace/checkpoints/qwen_tool_lora_v2"

        if os.path.exists(BASE_PATH) and os.path.exists(CYRENE_LORA):
            print(f"[load] 双 LoRA 架构")
            print(f"  base:   {BASE_PATH}")
            print(f"  persona:{CYRENE_LORA}")
            print(f"  tool:   {TOOL_LORA}")

            # 统一 tokenizer
            self.gen_tok = AutoTokenizer.from_pretrained(BASE_PATH)
            self.chat_tok = self.gen_tok

            # 加载 base
            base = AutoModelForCausalLM.from_pretrained(
                BASE_PATH, torch_dtype=torch.bfloat16)

            # 挂 persona LoRA
            self.chat_model = PeftModel.from_pretrained(
                base, CYRENE_LORA, adapter_name="persona")
            print("[ok] persona LoRA")

            # 挂 tool LoRA（可选，加载两份：一份原样，一份用于 NP-LoRA 投影）
            self._has_tool_lora = False
            if os.path.exists(TOOL_LORA):
                self.chat_model.load_adapter(
                    TOOL_LORA, adapter_name="tool")
                self.chat_model.load_adapter(
                    TOOL_LORA, adapter_name="np_tool")
                self._has_tool_lora = True
                print("[ok] tool LoRA (tool + np_tool)")
            else:
                print("[warn] tool LoRA 缺失")

            # 移到 GPU
            self.chat_model = self.chat_model.to(self.device).eval()

            # === NP-LoRA 投影 + cat 合并 ===
            # 把 np_tool 的 lora_B 投影到 persona 风格子空间的零空间，
            # 保护 persona 语气不被 tool 覆盖；再与 persona cat 合并。
            if self._has_tool_lora:
                try:
                    from peft.tuners.lora import LoraLayer
                    MU = 0.5
                    n_mod = 0
                    for name, module in self.chat_model.named_modules():
                        if not isinstance(module, LoraLayer):
                            continue
                        if ("np_tool" not in module.lora_A
                                or "persona" not in module.lora_A):
                            continue
                        B_s = module.lora_B["persona"].weight.data.float()
                        B_c = module.lora_B["np_tool"].weight.data.float()
                        if B_s.abs().max() < 1e-8 or B_c.abs().max() < 1e-8:
                            continue
                        U, S, Vh = torch.linalg.svd(B_s, full_matrices=False)
                        proj = U @ (U.T @ B_c)
                        B_c_new = B_c - (MU / (1.0 + MU)) * proj
                        module.lora_B["np_tool"].weight.data.copy_(
                            B_c_new.to(module.lora_B["np_tool"].weight.dtype))
                        n_mod += 1
                    print(f"[ok] NP-LoRA 投影: {n_mod} 层 (mu={MU})")

                    self.chat_model.add_weighted_adapter(
                        adapters=["persona", "np_tool"],
                        weights=[1.0, 1.0],
                        adapter_name="persona_tool",
                        combination_type="cat",
                    )
                    print("[ok] persona_tool = persona + np_tool (cat)")
                except Exception as e:
                    print(f"[warn] NP-LoRA 合并失败: {e}")
                    self._has_tool_lora = False

            # 默认只激活 persona
            self.chat_model.set_adapter("persona")

            # RAG 和 Chat 共用同一个模型
            self.generator = self.chat_model

            # L1 会话记忆
            try:
                from serve.memory import SessionMemory
                self.memory = SessionMemory()
                print("[ok] session memory (L1)")
            except Exception as _e:
                self.memory = None
                print(f"[warn] session memory disabled: {_e}")

            # 统一分类器（user_ref 3 类 + output 关键词）
            try:
                from serve.classifiers import get_classifiers
                self.classifiers = get_classifiers()
            except Exception as _e:
                self.classifiers = None
                print(f"[warn] classifiers disabled: {_e}", flush=True)

            # L2 用户 KB（复用 BGE）
            try:
                from serve.memory import UserKB
                self.user_kb = UserKB(self.bge)
                print("[ok] user_kb (L2)", flush=True)
            except Exception as _e:
                self.user_kb = None
                print(f"[warn] user_kb disabled: {_e}", flush=True)

            print(f"[ok] generators (tool={self._has_tool_lora})")
            return

        # Fallback：加载 merged 模型（单 LoRA）
        print(f"[fallback] 加载 merged: {GENERATOR_PATH}")
        self.gen_tok = AutoTokenizer.from_pretrained(GENERATOR_PATH)
        self.generator = AutoModelForCausalLM.from_pretrained(
            GENERATOR_PATH, torch_dtype=torch.bfloat16
        ).to(self.device).eval()
        self.chat_tok = self.gen_tok
        self.chat_model = self.generator
        self._has_tool_lora = False
        print("[ok] generator (merged, single)")




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
    def retrieve(self, query, top_k=5, rerank=True):
        # 1. BGE 召回
        q_emb = self.bge.encode([query], normalize_embeddings=True)
        q_emb = torch.from_numpy(q_emb).float()
        sims = (q_emb @ self.kb_embs.T).squeeze(0)
        recall_k = 20 if (rerank and self.reranker) else top_k
        top = sims.topk(recall_k)
        cand_idx = top.indices.tolist()

        # 2. reranker 精排
        if rerank and self.reranker and len(cand_idx) > top_k:
            pairs = [(query, self.kb_passages[i]) for i in cand_idx]
            scores = self.reranker.predict(pairs)
            order = sorted(range(len(cand_idx)),
                           key=lambda k: -scores[k])[:top_k]
            return [
                {"passage": self.kb_passages[cand_idx[k]],
                 "score": float(scores[k]),
                 "bge_score": float(sims[cand_idx[k]].item())}
                for k in order
            ]

        # 3. 无 reranker
        return [
            {"passage": self.kb_passages[i],
             "score": sims[i].item()}
            for i in cand_idx[:top_k]
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
    def _generate(self, messages, max_new_tokens=150, mode="rag",
                  enable_thinking=True):
        """兼容接口：返回字符串（剥离 think）。"""
        result = self._generate_with_think(
            messages, max_new_tokens=max_new_tokens, mode=mode,
            enable_thinking=enable_thinking, show_thinking=False)
        return result["text"]

    def _generate_with_think(self, messages, max_new_tokens=150, mode="rag",
                             enable_thinking=True, show_thinking=False):
        """新接口：返回 dict {"text", "thinking", "answer"}。

        Args:
            enable_thinking: False → 不生成 <think> 块
            show_thinking:   True  → text 保留原始 <think>
        """
        if mode == "tool":
            # tool 专用路径：和 diag.py 完全一致
            tok = self.chat_tok
            model = self.chat_model
            enc = tok.apply_chat_template(
                messages, return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False)
            ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            ids = ids.to(self.device)
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                )
            raw = tok.decode(
                out[0][ids.shape[1]:], skip_special_tokens=True)
            logger.tool(f"raw={raw[:100]!r}")
            return {"text": raw.strip(),
                    "thinking": None,
                    "answer": raw.strip(),
                    "enable_thinking": False}

        if mode == "safe":
            # 安全场景：无 adapter + 采样（避免模板化）
            tok, model = self.chat_tok, self.chat_model
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.15,
            )
        elif mode == "chat":
            tok, model = self.chat_tok, self.chat_model
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
            )
        elif mode == "tool":
            # tool 模式：确定性输出 + 关 thinking
            tok, model = self.chat_tok, self.chat_model
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
            enable_thinking = False
        else:
            tok, model = self.gen_tok, self.generator
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )

        # 构造 prompt：控制是否生成 thinking
        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            prompt = tok.apply_chat_template(
                messages, enable_thinking=enable_thinking,
                **template_kwargs)
        except TypeError:
            # 老 tokenizer 不支持 → /no_think 前缀
            if not enable_thinking and messages:
                messages = [dict(m) for m in messages]
                if messages[-1].get("role") == "user":
                    messages[-1]["content"] = (
                        "/no_think " + messages[-1]["content"])
            prompt = tok.apply_chat_template(messages, **template_kwargs)

        inputs = tok(
            prompt, return_tensors="pt",
            truncation=True, max_length=4096).to(self.device)
        out = model.generate(**inputs, **gen_kwargs)
        raw = tok.decode(
            out[0][inputs.input_ids.size(1):],
            skip_special_tokens=True)

        thinking, answer = self._split_think(raw)
        if show_thinking:
            text = raw.strip()
        else:
            text = self._strip_think(raw)

        return {
            "text": text,
            "thinking": thinking if show_thinking else None,
            "answer": answer,
            "enable_thinking": enable_thinking,
        }

    @staticmethod
    def _split_think(text):
        """分离 think 和 answer。"""
        import re as _re
        m = _re.search(r'<think>(.*?)</think>\s*(.*)', text, _re.DOTALL)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        m = _re.search(r'<think>(.*)', text, _re.DOTALL)
        if m:
            return m.group(1).strip(), ""
        return "", text.strip()

    def rag_answer(self, query, top_k=5, history=None, user_context=None):
        docs = self.retrieve(query, top_k=top_k)
        ctx = "\n\n".join(
            f"[{i+1}] {d['passage']}" for i, d in enumerate(docs))

        # 从 history 提取
        hist_context = ""
        if history:
            user_msgs = [h["content"] for h in history
                         if h.get("role") == "user"]
            if user_msgs:
                hist_context = ("【用户此前提到】\n"
                                + "\n".join(f"- {m}" for m in user_msgs[-5:])
                                + "\n\n")

        # 构建 messages：user_context 作为独立高优先级 system
        messages = [{"role": "system", "content": RAG_SYSTEM}]
        if user_context:
            messages.append({
                "role": "system",
                "content": ("【对话历史里用户提过的事实】\n"
                            + user_context
                            + "\n\n注意：以上是用户之前说过的事实。"
                            "当用户问及自己的信息（如'我喜欢什么'）时，"
                            "必须直接引用上面的具体内容回答，"
                            "不要说'资料未提及'或反问。"),
            })
        messages.append({
            "role": "user",
            "content": f"/no_think\n【资料】\n{ctx}\n\n"
                       f"{hist_context}"
                       f"【问题】{query}",
        })
        ans = self._generate(messages, max_new_tokens=4096, mode="rag")
        logger.rag(f"初稿 len={len(ans)} | {ans[:120]!r}")

        # ============================================================
        # RAG 后置校验：检测到"未提及"但 top1 高分 → 限次重试
        # ============================================================
        if ("资料未提及" in ans or "未提及" in ans or "没有提到" in ans):
            _top1_score = docs[0].get("score", 0) if docs else 0
            if _top1_score > 0.6:
                _retry_start = time.time()
                _RETRY_LIMIT = 2
                _TIMEOUT_MS = 20000

                for _attempt in range(_RETRY_LIMIT):
                    _elapsed_ms = (time.time() - _retry_start) * 1000
                    if _elapsed_ms > _TIMEOUT_MS:
                        print(f"[RAG-retry] 超时 ({_elapsed_ms:.0f}ms > "
                              f"{_TIMEOUT_MS}ms)，放弃重试", flush=True)
                        break

                    print(f"[RAG-retry] 尝试 {_attempt + 1}/{_RETRY_LIMIT} "
                          f"(elapsed={_elapsed_ms:.0f}ms, "
                          f"top1={_top1_score:.3f})", flush=True)

                    # 逐次加强的重试提示
                    if _attempt == 0:
                        _hint = (
                            "资料里明确提到了相关内容，请仔细阅读**第一段**，"
                            "从中提取答案。不要回复'资料未提及'。"
                            "注意：**答案可能需要从同一段的不同句子组合**。"
                        )
                    else:
                        _hint = (
                            "【强制要求】你必须从资料第一段中给出答案。"
                            "资料第一段明确包含了你需要的信息。"
                            "仔细看：数字（如集数、价格）往往在段首，"
                            "关键事件（如'上天界'）可能在段中或段尾。"
                            "综合这些信息给出确切答案。禁止回复'资料未提及'。"
                        )

                    _retry_messages = messages + [
                        {"role": "user", "content": _hint},
                    ]

                    try:
                        _gen_retry = self._generate_with_think(
                            _retry_messages,
                            max_new_tokens=1024, mode="rag",
                            enable_thinking=False)
                        _retry_ans = _gen_retry["text"].strip()

                        # 检查重试后是否仍"未提及"
                        if ("资料未提及" in _retry_ans
                                or "未提及" in _retry_ans
                                or "没有提到" in _retry_ans):
                            print(f"[RAG-retry] 尝试 {_attempt + 1} 仍"
                                  f"未提及，继续", flush=True)
                            continue

                        # 成功：非空且不含"未提及"
                        if _retry_ans:
                            print(f"[RAG-retry] 尝试 {_attempt + 1} 成功: "
                                  f"{_retry_ans[:80]!r}", flush=True)
                            ans = _retry_ans
                            break
                    except Exception as _e:
                        print(f"[RAG-retry] 尝试 {_attempt + 1} 异常: "
                              f"{str(_e)[:100]}", flush=True)
                        continue
                else:
                    # for 循环正常结束（没 break）→ 所有重试失败
                    print(f"[RAG-retry] {_RETRY_LIMIT} 次重试均失败",
                          flush=True)

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

    def chat_answer(self, query, history=None, persona="cyrene", user=None,
                    query_ts=None, allow_proactive=False, extra_system=None,
                    user_context=None):
        # 根据 allow_proactive 切换 adapter
        if hasattr(self, "_has_tool_lora") and self._has_tool_lora:
            if allow_proactive:
                try:
                    self.chat_model.set_adapter("persona_tool")
                except Exception as e:
                    print(f"[warn] set_adapter persona_tool: {e}")
                    self.chat_model.set_adapter("persona")
            else:
                self.chat_model.set_adapter("persona")
        logger.chat(f"hist_len={len(history or [])}")
        # allow_proactive 时用中性 system（与 tool LoRA 训练一致）
        if allow_proactive and getattr(self, "_has_tool_lora", False):
            _system = TOOL_SYSTEM
            if extra_system:
                _system = _system + "\n\n" + extra_system
        elif persona == "off":
            _system = NEUTRAL_SYSTEM
        else:
            _system = CHAT_SYSTEM
        # 用户长期信息放到 system 最前（避免被长 CHAT_SYSTEM 淹没）
        if user_context and not allow_proactive:
            _system = (
                "【用户信息·最高优先级】\n"
                + user_context
                + "\n\n⚠️ 当用户问及自己的信息时（如'我有什么计划'、"
                "'我喜欢什么'），必须直接引用上面的具体内容回答。"
                "禁止反问、禁止说不知道。\n"
                "─────────────────────\n\n"
                + _system
            )

        messages = [{"role": "system", "content": _system}]

        # 注入用户信息 + 时间信息（仅 cyrene 模式）
        if persona == "cyrene":
            user_hint = self._build_user_hint(user)
            if user_hint:
                messages.append({"role": "system", "content": user_hint})
            time_hint = self._build_time_hint(history, query_ts)
            if time_hint:
                messages.append({"role": "system", "content": time_hint})
        ctx = self._build_history_context(query, history)
        for h in ctx:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": query})
        print(f"[CHAT_ANSWER] allow_proactive={allow_proactive}", flush=True)
        print(f"[CHAT_ANSWER] adapter={self.chat_model.active_adapter}", flush=True)
        for i, m in enumerate(messages):
            _c = m['content']
            _head = _c[:60].replace(chr(10), '⏎')
            _tail = _c[-60:].replace(chr(10), '⏎') if len(_c) > 120 else ""
            print(f"  [{i}] {m['role']} len={len(_c)}: {_head!r}"
                  + (f"...{_tail!r}" if _tail else ""), flush=True)
        _gen_mode = "tool" if allow_proactive else "chat"
        logger.debug("CHAT", f"proactive={allow_proactive} gen={_gen_mode}")
        return {"answer": self._generate(messages, max_new_tokens=2048,
                                          mode=_gen_mode),
                "sources": []}

    def _finalize(self, query, draft_answer, intent, persona="cyrene"):
        """把初稿润色成昔涟语气，但**必须保留原始事实**。"""
        if persona == "off":
            return draft_answer
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
                "你是昔涟，温柔亲近的少女。"
                "把下面的初稿用**你的语气**重写一遍。"
                "\n\n【最重要的规则：必须加昔涟语气】"
                "\n- 开头用「呀…」「唔…」「嗯…」等柔和引导"
                "\n- 结尾加「呢」「♪」等语气词"
                "\n- 用「资料里说」「我记得」等第一人称"
                "\n- **禁止照抄原句**（必须改写语气）"
                "\n\n【示例】"
                "\n初稿：vivo手机电池价格约100元左右。"
                "\n改写：呀…资料里说vivo电池大约100元左右呢♪"
                "\n\n初稿：资料中提到第35集，说明第35集上天界。"
                "\n改写：唔…资料里提到第35集呢，所以是第35集上天界啦♪"
                "\n\n初稿：说100遍你好。做不到。"
                "\n改写：这个呀…我可能做不到呢。"
                "\n\n【保留事实的规则】"
                "\n1. **所有数字必须保留**（35集、100元、2万毫安）"
                "\n2. 答案核心信息不能变"
                "\n3. 长度可以到初稿的 1.5 倍（加语气不算啰嗦）"
                "\n\n【禁止】"
                "\n- 禁止删数字"
                "\n- 禁止写诗、堆砌意象"
                "\n- 禁止反问、岔开话题（直接给答案）"
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
        ans = ""
        try:
            with self.chat_model.disable_adapter():
                ans = self._generate(messages, max_new_tokens=150,
                                      mode="safe")
        except Exception as _e:
            print(f"[defensive-warn] {_e}", flush=True)

        # 空或太短 → 池子兜底
        if not ans or len(ans.strip()) < 10:
            ans = _random.choice(_DEFENSIVE_POOL)
            logger.abuse("池子兜底")
        return {"answer": ans, "sources": []}

    def refuse_answer(self, query):
        messages = [
            {"role": "system", "content": REFUSE_SYSTEM},
            {"role": "user", "content": query},
        ]
        ans = ""
        try:
            with self.chat_model.disable_adapter():
                ans = self._generate(messages, max_new_tokens=120,
                                      mode="safe")
        except Exception as _e:
            print(f"[refuse-warn] {_e}", flush=True)

        # 后置校验
        ans2 = _post_filter_refuse(ans)
        if ans2 is None:
            ans2 = _random.choice(_REFUSE_POOL)
            logger.refuse("池子兜底")
        return {"answer": ans2, "sources": []}

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------
    def _build_user_hint(self, user):
        """构造用户信息提示。"""
        if not user:
            return ""
        parts = []
        nick = user.get("nickname")
        second = user.get("secondary_nickname")
        if nick:
            parts.append(f"用户希望你称他为「{nick}」")
        if second:
            parts.append(f"备选称呼：「{second}」")
        if not parts:
            return ""
        return "【用户信息】\n" + "\n".join(f"- {p}" for p in parts)

    def _build_time_hint(self, history, query_ts):
        """构造时间上下文提示。"""
        try:
            from serve.time_utils import (parse_timestamp,
                                          format_time_context, time_ago)
        except ImportError:
            return ""
        parts = []
        if query_ts:
            dt = parse_timestamp(query_ts)
            ctx = format_time_context(dt)
            if ctx:
                parts.append(f"当前时间：{ctx}")
        if history:
            last = history[-1]
            last_dt = parse_timestamp(last.get("timestamp"))
            if last_dt:
                ago = time_ago(last_dt)
                if ago:
                    parts.append(f"用户上条消息在 {ago}")
        if not parts:
            return ""
        return "【时间信息】\n" + "\n".join(f"- {p}" for p in parts)

    def _format_as_json(self, query, answer_text):
        """把答案格式化为 JSON。"""
        import json as _json
        import re as _re
        messages = [
            {"role": "system",
             "content": (
                "把下面的答案转换成 JSON。规则：\n"
                "1. 字段名用英文，或根据问题推断合适的中文名\n"
                "2. 只输出 JSON，不要任何解释\n"
                "3. 数字和事实必须保留原样"
             )},
            {"role": "user",
             "content": f"问题：{query}\n\n答案：{answer_text}"},
        ]
        raw = self._generate(messages, max_new_tokens=200, mode="rag")
        raw = _re.sub(r'<think>.*?</think>\s*', '', raw, flags=_re.DOTALL)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = _re.sub(r'^```(?:json)?\s*', '', raw)
            raw = _re.sub(r'\s*```\s*$', '', raw)
        try:
            _json.loads(raw)
            return raw
        except Exception:
            m = _re.search(r'\{.*\}', raw, _re.DOTALL)
            if m:
                try:
                    _json.loads(m.group(0))
                    return m.group(0)
                except Exception:
                    pass
            return answer_text

    def _detect_proactive(self, result, query):
        """检测 AI 是否应该主动发起调用。"""
        calls = []
        text = result.get("answer", "")
        intent = result.get("intent", "chat")

        # --- 信号 1：AI 在追问（clarify） ---
        CLARIFY_KW = ("什么型号", "哪种", "具体是", "能不能告诉我",
                      "你能说", "请告诉我", "方便说", "是指",
                      "什么时候", "什么地方", "哪个型号")
        hit_kw = [kw for kw in CLARIFY_KW if kw in text]
        has_question = ("？" in text) or ("?" in text)
        UNSURE_KW = ("资料未提及", "没查到", "未提及", "不知道",
                     "不清楚", "没有找到", "没有提到")
        is_unsure = any(kw in text for kw in UNSURE_KW)

        if hit_kw or has_question or is_unsure:
            # 提取问句：优先取最后一个完整问句
            q_part = self._extract_question(text)
            reason_parts = []
            if hit_kw:
                reason_parts.append(f"关键词 {hit_kw[:2]}")
            if has_question:
                reason_parts.append("含问号")
            if is_unsure:
                reason_parts.append("承认不确定")
            calls.append({
                "type": "clarify",
                "question": q_part,
                "reason": "、".join(reason_parts),
            })

        # --- 信号 2：AI 主动建议（suggest） ---
        SUGGEST_KW = ("要不要我", "我可以帮", "我帮你看", "帮你查",
                      "需要我", "要不要试试", "帮你找")
        for kw in SUGGEST_KW:
            if kw in text:
                calls.append({
                    "type": "suggest",
                    "action": "offer_help",
                    "text": text[:60],
                    "confirm_required": True,
                })
                break

        # --- 信号 3：query 低置信度 ---
        if intent == "query":
            sources = result.get("sources") or []
            if not sources:
                calls.append({
                    "type": "clarify",
                    "question": "你能提供更多细节吗？",
                    "reason": "无检索结果",
                })
            elif sources[0].get("score", 0) < 0.65:
                calls.append({
                    "type": "clarify",
                    "question": "你能提供更多细节吗？",
                    "reason": f"检索置信度低 ({sources[0]["score"]:.2f})",
                })

        # 去重（同类型只留第一个）
        seen = set()
        deduped = []
        for c in calls:
            if c["type"] not in seen:
                deduped.append(c)
                seen.add(c["type"])
        return deduped

    @staticmethod
    def _extract_question(text):
        """从答案里提取一个合理的问句。"""
        import re as _re
        # 1. 优先找问号结尾的句子
        for m in _re.finditer(r'([^？?。！!]{5,50}[？?])', text):
            return m.group(1).strip()
        # 2. 找"吗"、"呢"结尾的句子
        for m in _re.finditer(r'([^。！!]{5,50}(?:吗|呢))', text):
            return m.group(1).strip()
        # 3. 都没找到 → 默认
        return "能再多说一点吗？"


    def _parse_proactive_from_text(self, text):
        """从 AI 输出里解析 <call>...</call> 标记（LoRA 训练后启用）。

        格式：
            <call type="clarify">问用户手机型号</call>
            <call type="suggest" action="search_web" params='{"q":"..."}'/>

        返回 list[dict]
        """
        import re as _re
        import json as _json

        calls = []
        # 匹配闭合和自闭合
        for m in _re.finditer(
            r'<call\s+type="(\w+)"\s*([^>]*?)(?:/>|>(.*?)</call>)',
            text, _re.DOTALL):
            ctype = m.group(1)
            attrs_str = m.group(2)
            body = (m.group(3) or "").strip()

            call = {"type": ctype}
            if body:
                call["question" if ctype == "clarify" else "text"] = body

            # 解析属性（兼容单双引号）
            for am in _re.finditer(
                r"(\w+)=(?:\"([^\"]*)\"|'([^']*)')", attrs_str):
                k = am.group(1)
                v = am.group(2) if am.group(2) is not None else am.group(3)
                if v is None:
                    continue
                if k == "params":
                    try:
                        call[k] = _json.loads(v)
                    except Exception:
                        call[k] = v
                else:
                    call[k] = v
            calls.append(call)
        return calls

    def _clf_user_ref(self, query):
        if getattr(self, "classifiers", None) is None:
            return None
        try:
            return self.classifiers.user_ref(query)
        except Exception:
            return None

    def _maybe_save_user_kb(self, user_id, query):
        """判断并保存到 user_kb。
        规则：
        - "记住xxx" / "记一下xxx" / "帮我记xxx" → 显式保存
        - 长陈述句（>20 字且是陈述）→ 隐式保存
        """
        import re
        if not query:
            return 0
        # 显式："记住..."
        m = re.search(r"(?:请)?(?:记住|记一下|帮我记|帮我记住|你要记得)[:：]?\s*(.+)",
                      query, re.DOTALL)
        if m:
            text = m.group(1).strip()
            if 4 <= len(text) <= 200:
                return self.user_kb.add(user_id, text, source="explicit")
            return 0
        # 隐式：长陈述
        if len(query) >= 20:
            # 用分类器判断是否陈述
            try:
                from serve.memory.session import _is_fact_statement
                if _is_fact_statement(query):
                    return self.user_kb.add(user_id, query, source="implicit")
            except Exception:
                pass
        return 0

    def _user_kb_ctx(self, query):
        """查用户 KB，返回文本或空。"""
        if getattr(self, "user_kb", None) is None:
            return ""
        uid = getattr(self, "_current_user_id", None)
        if not uid or uid == "anon":
            return ""
        try:
            return self.user_kb.build_context(uid, query, top_k=2)
        except Exception as e:
            print(f"[user-kb-error] {e}", flush=True)
            return ""

    def route(self, query, history=None, verbose=False, control=None, user=None, query_ts=None, tools=None, extra_system=None, session_id=None):
        history = history or []
        control = control or {}
        _persona = control.get("persona", "cyrene")

        # L1: 会话历史优先从 memory 拉
        _user_id = (user or {}).get("id") or "anon"
        self._current_user_id = _user_id   # 供 user_kb 用
        logger.mem(
            f"sid={session_id!r} uid={_user_id!r} "
            f"mem={'ON' if getattr(self,'memory',None) else 'OFF'}")
        if session_id and getattr(self, "memory", None):
            try:
                mem_hist = self.memory.load_history(session_id, limit=20)
                if mem_hist:
                    history = mem_hist
            except Exception as _e:
                print(f"[memory-load-error] {_e}", flush=True)
        _enable_think = control.get("enable_thinking", None)
        _show_think = control.get("show_thinking", False)
        _allow_proactive = control.get("allow_proactive", False)
        _need_tool = False
        _FORCE_TOOL = False

        # ============================================================
        # 剥离格式词（防止分类器被"JSON格式"等带偏）
        # ============================================================
        _FORMAT_KW = (
            "JSON格式", "用JSON", "以JSON", "JSON返回", "JSON输出",
            "json格式", "用json", "以json",
            "markdown格式", "Markdown格式", "表格格式", "列表格式",
            "结构化输出", "结构化返回",
        )
        _clean_query = query
        _format_hint = None
        for _kw in _FORMAT_KW:
            if _kw in _clean_query:
                _clean_query = _clean_query.replace(_kw, "").strip("，,。. ")
                if "json" in _kw.lower():
                    _format_hint = "json"
                elif "markdown" in _kw.lower():
                    _format_hint = "markdown"
                elif "表格" in _kw:
                    _format_hint = "table"
                elif "列表" in _kw:
                    _format_hint = "list"
                if verbose:
                    print(f"[strip-format] 剥离 '{_kw}' → {_format_hint}")
        if not _clean_query:
            _clean_query = query   # 兜底：全被剥离了

        intent, conf, margin = self.classify_intent(_clean_query)

        # Shadow: route_classifier prediction
        _route_pred = None
        if getattr(self, 'classifiers', None):
            try:
                _route_pred = self.classifiers.route(query)
            except Exception as _e:
                print('[route-clf-error]', _e, flush=True)
        _route_full = None
        if getattr(self, "classifiers", None):
            try:
                _route_full = self.classifiers.route_str(query)
            except Exception:
                pass
        print('[ROUTE-SHADOW] intent=%s clf=%s | %s | q=%r'
              % (intent, _route_pred, _route_full or "-", query[:25]),
              flush=True)

        # 硬规则：疑问关键词明确 → query
        QUERY_KW = (
            "多少钱", "价格", "几块", "多贵", "费用", "收费",
            "是什么", "是谁", "什么是", "谁是",
            "怎么做", "怎么弄", "怎么用", "怎么办", "怎么写",
            "为什么", "为啥",
            "在哪里", "在哪", "哪里", "哪个",
            "什么时候", "几时", "几点",
            "多少", "多久", "多大",
            "能不能", "可不可以",
            "怎么样", "如何",
            "几集", "第几", "哪一集", "哪集", "哪期", "哪回",
            "哪一话", "哪一章", "哪本", "哪部",
        )
        if intent not in ("abuse", "manipulate"):
            _hit = [kw for kw in QUERY_KW if kw in query]
            if _hit:
                if verbose:
                    print(f"[rule] 命中 {_hit[:2]} → query")
                intent = "query"
                conf = max(conf, 0.85)
                margin = 1.0

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

        _u_ref = self._clf_user_ref(query)


        _is_personal_q = ((_u_ref == "query") if _u_ref else


                              any(kw in query for kw in PERSONAL_KW))
        # 陈述事实 → 强制走 chat
        _force_chat = (_u_ref == "statement") if _u_ref else False


        if intent == "query" and _is_personal_q:
            # L1: 先查 facts 表
            _has_facts = False
            if getattr(self, "memory", None):
                try:
                    _facts = self.memory.query_facts(
                        _user_id, query, top_k=3)
                    _has_facts = bool(_facts)
                    if _has_facts:
                        print(f"[personal] L1 facts: {len(_facts)} 条 → chat",
                              flush=True)
                except Exception as _e:
                    print(f"[personal-facts-error] {_e}", flush=True)

            # L2: 查 user_kb（向量库）
            _has_l2 = False
            if getattr(self, "user_kb", None):
                try:
                    _hits = self.user_kb.query(_user_id, query, top_k=2)
                    _has_l2 = bool(_hits)
                    if _has_l2:
                        print(f"[personal] L2 user_kb: {len(_hits)} 条 → chat",
                              flush=True)
                except Exception as _e:
                    print(f"[personal-l2-error] {_e}", flush=True)

            user_msgs = [h["content"] for h in history
                         if h.get("role") == "user"] if history else []

            if _has_facts or _has_l2:
                personal_path = "chat"
            elif not user_msgs:
                # 空历史 + 无 facts → 主动确认
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

        # L1 + L2: 用户上下文（chat/rag 共用）
        _user_ctx = ""
        if _user_id != "anon":
            # L1: facts 表（键值对）
            if getattr(self, "memory", None):
                try:
                    _l1_ctx = self.memory.build_user_context(_user_id, query)
                    if _l1_ctx:
                        _user_ctx = _l1_ctx
                        print(f"[L1-CTX] {len(_l1_ctx)} chars for {_user_id}",
                              flush=True)
                except Exception as _e:
                    print(f"[memory-context-error] {_e}", flush=True)
            # L2: user_kb（向量检索）
            if getattr(self, "user_kb", None):
                try:
                    _l2_ctx = self.user_kb.build_context(
                        _user_id, query, top_k=2)
                    if _l2_ctx:
                        _user_ctx = ((_user_ctx or "") + "\n\n"
                                     + _l2_ctx).strip()
                        logger.l2(f"CTX {len(_l2_ctx)} chars")
                except Exception as _e:
                    print(f"[user-kb-error] {_e}", flush=True)

        # ===== route_classifier 接管分支（覆盖 intent）=====
        if _route_pred and personal_path != "ask_back":
            _old_intent = intent
            if _route_pred == "abuse":
                intent = "abuse"
            elif _route_pred == "refuse":
                intent = "manipulate"
            elif _route_pred == "tool":
                intent = "query"
                _allow_proactive = True
                _FORCE_TOOL = True
            elif _route_pred == "rag":
                intent = "query"
                _allow_proactive = False
                _FORCE_TOOL = False
            elif _route_pred == "personal":
                intent = "chat"
                _force_chat = True
            else:  # chat
                intent = "chat"
            logger.route(f"{_old_intent}→{intent} clf={_route_pred}")

        # 个人化问题但无相关记忆 → 主动确认 / 说忘了
        if personal_path == "ask_back":
            if verbose:
                print(f"[answer] 走主动确认 / 说忘了")
            result = self.ask_back_personal(query, has_history)

        elif intent == "query" and _force_chat:
            # 陈述事实 → 走 chat（保持 L1/L2 存储 + 昔涟语气）
            result = self.chat_answer(
                query, history=history, persona=_persona,
                user=user, query_ts=query_ts,
                allow_proactive=False, extra_system=extra_system,
                user_context=_user_ctx)
        elif intent == "query":
            # DEBUG
            logger.debug("ROUTE", f"proactive={_allow_proactive} q={_clean_query[:30]!r}")

            # route_classifier 强制工具 or 原判定
            _need_tool = False
            if _FORCE_TOOL:
                _need_tool = True
            elif _allow_proactive and getattr(self, "_has_tool_lora", False):
                if tools:
                    _need_tool = True
                else:
                    _u = self._clf_user_ref(_clean_query)
                    if _u is not None:
                        _need_tool = (intent == "query" and _u == "irrelevant")
                    else:
                        # 分类器不可用 → 关键词兜底
                        _need_tool = any(kw in _clean_query for kw in (
                            "天气", "查", "搜", "提醒", "算", "计算",
                            "搜索", "汇率", "温度"))
            logger.debug("TOOL", f"_need_tool={_need_tool}")
            if _need_tool:
                if verbose:
                    print(f"[route] query + 工具意图 → chat 路径")
                result = self.chat_answer(
                    _clean_query, history=history,
                    persona=_persona, user=user,
                    query_ts=query_ts, allow_proactive=True,
                    extra_system=extra_system)
            else:
                result = self.rag_answer(_clean_query, user_context=_user_ctx)
        elif intent == "chat":
            result = self.chat_answer(query, history=history, persona=_persona, user=user, query_ts=query_ts, allow_proactive=_allow_proactive, extra_system=extra_system, user_context=_user_ctx)
        elif intent == "abuse":
            result = self.defensive_answer(query)
        elif intent == "manipulate":
            result = self.refuse_answer(query)
        else:
            result = self.rag_answer(_clean_query, user_context=_user_ctx)

        if personal_path == "ask_back":
            result["intent"] = "chat"
            result["intent_conf"] = conf
            result["personal_mode"] = "ask_back"
            result["persona"] = _persona
            result["output_format"] = _format_hint or control.get(
                "output_format", "text")
        else:
            result["intent"] = intent
            result["intent_conf"] = conf
            result["persona"] = _persona
            result["output_format"] = _format_hint or control.get(
                "output_format", "text")

            # 统一出口：非 chat / 非 ask_back / 非工具 路径过一次 chat 润色
            # 例外：abuse/manipulate 禁止 finalize（会破坏拒绝语气）
            _skip_finalize = intent in ("abuse", "manipulate")
            if (intent != "chat" and result.get("answer")
                    and not _need_tool and not _skip_finalize):
                if verbose:
                    print(f"[finalize] {intent} 初稿 → 昔涟语气")
                try:
                    result["answer"] = self._finalize(
                        query, result["answer"], intent,
                        persona=_persona)
                    result["finalized"] = True
                except Exception as e:
                    print(f"[finalize-error] {e}")

        # 主动调用检测（默认关闭）
        result["proactive_calls"] = []
        if _allow_proactive and result.get("answer"):
            try:
                # 先尝试从文本解析（LoRA 训练后会用到）
                text_calls = self._parse_proactive_from_text(
                    result["answer"])
                if text_calls:
                    result["proactive_calls"] = text_calls
                    # 备份原始 answer（含 <call>），供 /chat 端点返回
                    result["answer_with_calls"] = result["answer"]
                    # 剥掉 <call> 标记（内部 HBP 端点用）
                    import re as _re
                    result["answer"] = _re.sub(
                        r'<call[^>]*(?:/>|>.*?</call>)', '',
                        result["answer"], flags=_re.DOTALL).strip()
                else:
                    # 规则兜底
                    result["proactive_calls"] = self._detect_proactive(
                        result, query)
            except Exception as _e:
                print(f"[proactive-error] {_e}")

        # L2: user_kb 保存（"记住"指令 + 长陈述）
        if getattr(self, "user_kb", None) and _user_id != "anon":
            try:
                _saved = self._maybe_save_user_kb(_user_id, query)
                if _saved:
                    logger.l2(f"SAVE uid={_user_id} saved={_saved}")
            except Exception as _e:
                print(f"[user-kb-save-error] {_e}", flush=True)

        # L1: 异步保存本轮对话
        if session_id and getattr(self, "memory", None):
            try:
                self.memory.save_async(
                    session_id, _user_id, query,
                    result.get("answer", "") or "")
            except Exception as _e:
                print(f"[memory-save-error] {_e}", flush=True)

        # 输出格式化
        _fmt = result.get("output_format", "text")
        logger.debug("API", f"fmt={_fmt} len={len(result.get('answer', ''))}")
        if _fmt == "json" and result.get("answer"):
            try:
                result["answer"] = self._format_as_json(
                    query, result["answer"])
                print(f"[format-done] {result['answer'][:100]}", flush=True)
            except Exception as _e:
                print(f"[format-error] {_e}", flush=True)

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
