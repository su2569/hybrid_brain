"""L2 用户 KB：向量库存储用户私有知识。

与通用 RAG 的区别：
- 通用 RAG：全局知识（12648 条，共享）
- 用户 KB：每用户私有（长文本/事件/领域知识）

与 L1 facts 的区别：
- L1 facts：结构化键值对（偏好/位置/身份）
- L2 user_kb：非结构化文本（长对话摘要、用户上传文档、事件细节）
"""
import os
import sqlite3
import threading
import time
import pickle
from typing import List, Dict, Optional


class UserKB:
    """用户私有向量库。用小矩阵 numpy 检索（<10k 条/用户不需要 FAISS）。"""

    def __init__(self, bge, db_path: str = "data/user_kb.db"):
        self.bge = bge
        self.db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _conn(self):
        c = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _init_db(self):
        conn = self._conn()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_kb (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            text TEXT NOT NULL,
            emb BLOB NOT NULL,
            source TEXT DEFAULT 'chat',
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_user ON user_kb(user_id, created_at DESC);
        """)
        conn.commit()
        conn.close()

    def add(self, user_id: str, text: str, source: str = "chat"):
        """添加用户知识。"""
        if not user_id or user_id == "anon" or not text or len(text) < 5:
            return 0
        emb = self.bge.encode([text], normalize_embeddings=True)[0]
        blob = pickle.dumps(emb.astype("float32"))
        with self._lock:
            conn = self._conn()
            # 去重：同 user 同 text 不重复写
            exists = conn.execute(
                "SELECT 1 FROM user_kb WHERE user_id=? AND text=? LIMIT 1",
                (user_id, text),
            ).fetchone()
            if exists:
                conn.close()
                return 0
            conn.execute(
                "INSERT INTO user_kb (user_id, text, emb, source, created_at) "
                "VALUES (?,?,?,?,?)",
                (user_id, text, blob, source, int(time.time())),
            )
            conn.commit()
            conn.close()
        return 1

    def query(self, user_id: str, question: str,
              top_k: int = 3, min_score: float = 0.35) -> List[Dict]:
        """检索用户知识。"""
        import numpy as np
        if not user_id or user_id == "anon" or not question:
            return []
        conn = self._conn()
        rows = conn.execute(
            "SELECT text, emb FROM user_kb WHERE user_id=? "
            "ORDER BY created_at DESC LIMIT 500",
            (user_id,),
        ).fetchall()
        conn.close()
        if not rows:
            return []
        texts, embs = [], []
        for t, b in rows:
            try:
                embs.append(pickle.loads(b))
                texts.append(t)
            except Exception:
                continue
        if not embs:
            return []
        mat = np.stack(embs)  # (N, D)
        q = self.bge.encode([question], normalize_embeddings=True)[0]
        sims = mat @ q
        order = sims.argsort()[::-1][:top_k]
        out = []
        for i in order:
            s = float(sims[i])
            if s >= min_score:
                out.append({"text": texts[i], "score": s})
        return out

    def build_context(self, user_id: str, question: str,
                      top_k: int = 2) -> str:
        """构造注入 system 的文本。无则返回空。"""
        hits = self.query(user_id, question, top_k=top_k)
        if not hits:
            return ""
        # 只返回列表，标题由外层统一加
        return "\n".join(f"· {h['text']}" for h in hits)

    def stats(self, user_id: str) -> Dict:
        conn = self._conn()
        n = conn.execute(
            "SELECT count(*) FROM user_kb WHERE user_id=?",
            (user_id,)
        ).fetchone()[0]
        conn.close()
        return {"user_id": user_id, "count": n}


# ---------- 单例 ----------
_KB_INSTANCE = None


def get_user_kb(bge=None) -> Optional[UserKB]:
    global _KB_INSTANCE
    if _KB_INSTANCE is None:
        if bge is None:
            return None
        _KB_INSTANCE = UserKB(bge)
    return _KB_INSTANCE
