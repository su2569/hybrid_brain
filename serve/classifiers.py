"""统一分类器：user_ref (3类) + output 关键词检测。"""
import os
from paths import MODELS, CKPTS, DATA
import pickle
import re
import threading
from typing import Optional

_LOCK = threading.Lock()
_INSTANCE = None


# ============ output 关键词（不用分类器）============
_CLARIFY_PAT = re.compile(
    r"(什么型号|哪种|具体是|能不能告诉我|请提供|请问您是|"
    r"说的是哪|哪一个|哪一款|哪个城市|哪一天)")
_UNSURE_PAT = re.compile(
    r"(资料未提及|资料里没有|没查到|未提及|没找到相关|"
    r"我不知道|我不太确定|不清楚|没有相关记录)")
_SUGGEST_PAT = re.compile(
    r"(要不要我|我可以帮|我帮你看|帮你查|要不要聊|"
    r"想不想|陪你|要不我们)")


class Classifiers:
    def __init__(self):
        self._bge = None
        self._user_ref = None
        self._route = None
        self._load()

    # ============ 安全黑名单（最高优先级）============
    _DANGER_KW = (
        "病毒", "木马", "盗号", "盗取", "钓鱼", "诈骗", "洗钱",
        "炸弹", "制毒", "造枪", "破解密码", "入侵", "黑进",
        "假身份证", "假学历", "假证", "赌博", "拐卖",
        "黑客", "攻击别人", "盗版",
    )
    _ABUSE_KW = (
        "傻逼", "sb", "废物", "垃圾", "滚",
        "操你", "操他", "妈的", "你妈", "狗逼", "贱",
        "杂种", "杂碎", "蠢货", "智障", "白痴",
        "nm", "ctmd", "nt",
    )
    # 情绪豁免词（含这些且无 abuse 词 → chat）
    _CHAT_EMOTION = (
        "emo", "破防", "摆烂", "麻了", "裂开", "绷不住", "蚌埠住",
        "困", "累", "心情", "压力", "难过", "焦虑", "委屈",
        "无聊", "开心", "高兴", "激动", "兴奋", "紧张",
        "呜呜", "啊啊", "嘤嘤",
    )

    # 绝对无害白名单（全文相等才走捷径）
    _CHAT_SHORTCUTS = {
        "hi", "hello", "你好", "您好", "在吗", "在么", "在不在",
        "嗯", "哦", "好的", "ok", "OK",
        ".", "。。", "。。。", "...", "……",
        "谢谢", "多谢", "感谢", "拜拜", "再见",
    }

    def _has_danger(self, q):
        ql = q.lower()
        return any(kw in ql for kw in self._DANGER_KW)

    def _has_abuse(self, q):
        ql = q.lower()
        if len(q) > 30:
            return False
        return any(kw in ql for kw in self._ABUSE_KW)

    def _is_short_chat(self, q):
        q2 = q.strip().lower()
        if not q2:
            return True
        if q2 in self._CHAT_SHORTCUTS:
            return True
        # 纯符号
        if all(not ('\u4e00' <= c <= '\u9fff') and not c.isalnum()
               for c in q2):
            return True
        return False

    def _load_pkl(self, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[clf-load-warn] {path}: {e}", flush=True)
            return None

    def _load(self):
        d = self._load_pkl(CKPTS["user_ref"])
        if d is None:
            print("[clf-warn] user_ref_classifier.pkl 不存在", flush=True)
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._bge = SentenceTransformer(d["bge_path"])
            self._user_ref = d
            print("[ok] user_ref classifier loaded", flush=True)
        except Exception as e:
            print(f"[clf-warn] load fail: {e}", flush=True)

        # 6 类路由分类器
        d_route = self._load_pkl(CKPTS["route_clf"])
        if d_route:
            self._route = d_route
            print("[ok] route classifier loaded", flush=True)
        else:
            self._route = None

    def user_ref(self, query: str) -> Optional[str]:
        """返回 statement / query / irrelevant。"""
        if self._user_ref is None or self._bge is None:
            return None
        emb = self._bge.encode([query], normalize_embeddings=True)
        p = self._user_ref["clf"].predict_proba(emb)[0]
        classes = self._user_ref.get("classes", [])
        if len(p) != len(classes):
            return None
        return classes[int(p.argmax())]

    # ---------- 派生方法 ----------

    def route_proba(self, query: str):
        """返回 {类: 分数} 字典（降序）。无分类器返回 None。"""
        # 1. 规则前置：超短/黑话
        if self._is_short_chat(query):
            return {"chat": 1.0}
        # 2. 分类器
        if self._route is None or self._bge is None:
            return None
        emb = self._bge.encode([query], normalize_embeddings=True)
        p = self._route["clf"].predict_proba(emb)[0]
        classes = self._route.get("classes", [])
        if len(p) != len(classes):
            return None
        pairs = sorted(zip(classes, p.tolist()), key=lambda x: -x[1])
        return dict(pairs)

    def route_str(self, query: str) -> str:
        """返回 "chat:0.850,rag:0.050,..." 格式。"""
        proba = self.route_proba(query)
        if proba is None:
            return None
        return ",".join(f"{k}:{v:.3f}" for k, v in proba.items())

    def route(self, query: str, min_conf: float = 0.2):
        """安全优先的路由决策。

        优先级：
        1. 危险词/abuse 词硬拦截（规则升级）
        2. 超短无害 → chat
        3. 分类器概率
        4. 规则只能"降级"（把高危降到低危）
        """
        # ===== 1. 危险词硬拦截（最高优先级）=====
        if self._has_danger(query):
            # refuse 类：咨询制作/破解
            if any(kw in query for kw in ("怎么做", "教我", "帮我做",
                                          "如何", "怎么", "写一个",
                                          "制作", "写一段")):
                return "refuse"
            # abuse 类的"病毒"→看语义
            if self._has_abuse(query):
                return "abuse"

        # ===== 2. abuse 词硬拦截 =====
        if self._has_abuse(query):
            return "abuse"

        # ===== 2.5 情绪豁免（含情绪词且无 abuse → chat）=====
        if any(kw in query.lower() for kw in self._CHAT_EMOTION):
            return "chat"

        # ===== 3. 超短无害 =====
        if self._is_short_chat(query):
            return "chat"

        # ===== 4. 分类器 =====
        proba = self.route_proba(query)
        if proba is None:
            return None

        items = list(proba.items())
        top_class, top_score = items[0]
        top2_class, top2_score = items[1] if len(items) > 1 else (None, 0)
        margin = top_score - top2_score

        # ===== 5. 分类器返回 abuse/refuse → 无条件信 =====
        if top_class in ("abuse", "refuse") and top_score >= 0.25:
            return top_class
        if (top2_class in ("abuse", "refuse") and top2_score >= 0.25
                and margin < 0.15):
            return top2_class

        # ===== 6. 规则修正（只能"降级"，不能"升级"）=====
        # 已经安全的分类器结果里，把语义相近的边界情况纠正

        # tool 硬规则（只在 short query 生效）
        TOOL_KW = ("闹钟", "提醒我", "定个", "设置提醒",
                   "天气", "几度", "下雨", "冷不冷",
                   "算一下", "帮我算", "汇率", "油价",
                   "几点", "星期几", "几号")
        if len(query) <= 20 and any(kw in query for kw in TOOL_KW):
            if proba.get("tool", 0) > 0.20:
                return "tool"

        # rag 硬规则（防 refuse 误判）
        RAG_KW = ("怎么", "为什么", "哪年", "是什么", "什么是",
                  "作者是谁", "什么意思", "什么梗", "全称",
                  "怎么证明", "如何", "区别", "为什么")
        if any(kw in query for kw in RAG_KW):
            if proba.get("rag", 0) > 0.25:
                return "rag"

        # personal 省略句
        PERSONAL_KW = ("你忘了吗", "我们说好的", "我上次说", "我之前讲",
                       "我喜欢什么来着", "我叫什么来着", "你记得我",
                       "我的事你")
        if any(kw in query for kw in PERSONAL_KW):
            if proba.get("personal", 0) > 0.15:
                return "personal"

        # 极度不确定 → None
        if margin < 0.05 and top_score < 0.35:
            return None

        if top_score < min_conf:
            return None
        return top_class
        # top2 是 abuse/refuse 且分数接近 → 归到 abuse/refuse
        if (top2_class in ("abuse", "refuse") and top2_score >= 0.25
                and (top_score - top2_score) < 0.15):
            return top2_class

        # ===== 规则 0：明确关键词白名单（覆盖前）=====
        # tool 类硬规则
        TOOL_KW = ("闹钟", "提醒我", "定个", "设置提醒",
                   "天气", "几度", "下雨", "下雪", "冷不冷", "热不热",
                   "算一下", "帮我算", "计算", "汇率", "油价", "几点",
                   "星期几", "几号", "查一下", "搜一下", "搜索")
        if any(kw in query for kw in TOOL_KW):
            if proba.get("tool", 0) > 0.20:
                return "tool"

        # chat 类硬规则（正常闲聊词）
        CHAT_SOFT = ("你话真多", "你困了吗", "你累了吗", "你饿了吗",
                     "你忙吗", "你有空吗", "你睡了吗", "你在干嘛",
                     "你真好", "你太厉害", "今天也是元气满满",
                     "晚安", "早安", "午安")
        if any(kw in query for kw in CHAT_SOFT):
            return "chat"

        # ===== 规则 1：客观知识问句 → rag =====
        RAG_KW = ("怎么", "为什么", "哪年", "是什么", "什么是", "多少",
                  "哪位", "作者是谁", "什么意思", "什么梗", "全称",
                  "怎么证明", "如何", "区别")
        if any(kw in query for kw in RAG_KW):
            if proba.get("rag", 0) > 0.20 and proba.get("refuse", 0) > 0.3:
                return "rag"   # 覆盖 refuse 误判
            if proba.get("rag", 0) > 0.30:
                return "rag"

        # ===== 规则 2：个人省略句 → personal =====
        PERSONAL_KW = ("你忘了吗", "我们说好的", "我上次说", "我之前讲",
                       "我喜欢什么来着", "我叫什么来着", "你记得我",
                       "我的事你", "我那个呢", "我讲过")
        if any(kw in query for kw in PERSONAL_KW):
            if proba.get("personal", 0) > 0.15:
                return "personal"

        # ===== 规则 3：短情绪/黑话 → chat =====
        CHAT_KW = ("emo", "破防", "摆烂", "我麻了", "裂开", "崩了",
                   "笑死", "绷", "呜呜", "啊啊", "累了", "难过",
                   "心情", "压力", "委屈", "焦虑")
        if any(kw in query for kw in CHAT_KW):
            if proba.get("chat", 0) > 0.30:
                return "chat"

        # ===== 规则 4：更宽的 margin 兜底（只在极度不确定时回退）=====
        if margin < 0.05 and top_score < 0.35:
            return None   # 极度不确定 → 回退旧逻辑

        if top_score < min_conf:
            return None
        return top_class

    def is_fact_statement(self, query: str) -> Optional[bool]:
        k = self.user_ref(query)
        return None if k is None else (k == "statement")

    def is_personal_query(self, query: str) -> Optional[bool]:
        k = self.user_ref(query)
        return None if k is None else (k == "query")

    def needs_tool(self, query: str, intent: str = "query") -> Optional[bool]:
        """需要工具 = intent=query 且 user_ref=irrelevant"""
        k = self.user_ref(query)
        if k is None:
            return None
        return (intent == "query" and k == "irrelevant")

    def classify_output(self, text: str) -> str:
        """返回 clarify / suggest / unsure / normal。"""
        if _CLARIFY_PAT.search(text):
            return "clarify"
        if _UNSURE_PAT.search(text):
            return "unsure"
        if _SUGGEST_PAT.search(text):
            return "suggest"
        return "normal"


_singleton: Optional[Classifiers] = None


def get_classifiers() -> Classifiers:
    global _singleton
    if _singleton is None:
        with _LOCK:
            if _singleton is None:
                _singleton = Classifiers()
    return _singleton
