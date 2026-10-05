"""统一分类器：user_ref (3类) + output 关键词检测。"""
import os
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
        self._load()

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
        d = self._load_pkl("checkpoints/user_ref_classifier.pkl")
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
