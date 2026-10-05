"""L1 会话记忆：SQLite 存储 + 规则事实抽取 + 上下文注入。

不碰模型权重，删表即回滚。
"""
import os
import pickle
import re
import sqlite3
import threading
import time
from typing import List, Dict, Optional


# --------- 规则事实抽取 ---------
_FACT_RULES = [
    # 偏好
    (r"我(?:很|特别|超)?(?:喜欢|爱|常喝|常吃|常去)([^，。！？,!?\n]{2,30})", "偏好"),
    (r"我(?:不喜欢|讨厌|不爱|最烦)([^，。！？,!?\n]{2,30})", "反偏好"),
    # 身份
    (r"我(?:是|叫)([^，。！？,!?\n]{2,20})", "身份"),
    (r"我的名字(?:是|叫)([^，。！？,!?\n]{2,20})", "姓名"),
    # 位置
    (r"我(?:住在|在|住在)([^，。！？,!?\n]{2,30})", "位置"),
    # 属性
    (r"我(?:一般|平时|通常)([^，。！？,!?\n]{2,30})", "习惯"),
    # 明确指令
    (r"(?:请)?(?:记住|记一下|帮我记)[:：]?\s*([^，。！？,!?\n]{4,60})", "显式记忆"),
]

# 命中这些词才考虑注入 facts（避免所有回复被污染）
_PERSONAL_TRIGGERS = [
    "我", "自己", "之前", "上次", "记得", "我说过", "我的",
    "喜欢", "讨厌", "习惯", "偏好",
]


_FC_SINGLETON = {"model": None, "bge": None, "loaded": False}


def _load_fact_classifier():
    """加载 BGE + LR 分类器（单例）。"""
    if _FC_SINGLETON["loaded"]:
        return _FC_SINGLETON
    _FC_SINGLETON["loaded"] = True
    try:
        from sentence_transformers import SentenceTransformer
        ckpt = "checkpoints/fact_classifier.pkl"
        if not os.path.exists(ckpt):
            print("[warn] fact_classifier.pkl 不存在，退回规则", flush=True)
            return _FC_SINGLETON
        with open(ckpt, "rb") as f:
            d = pickle.load(f)
        _FC_SINGLETON["model"] = d["clf"]
        _FC_SINGLETON["bge"] = SentenceTransformer(d["bge_path"])
        print("[ok] fact classifier loaded", flush=True)
    except Exception as e:
        print(f"[warn] fact classifier disabled: {e}", flush=True)
    return _FC_SINGLETON


def _is_fact_statement(query: str) -> bool:
    """走共享 user_ref 分类器；不可用时退回疑问词。"""
    try:
        from serve.classifiers import get_classifiers
        clf = get_classifiers()
        p = clf.is_fact_statement(query)
        if p is not None:
            return p
    except Exception:
        pass
    _Q = ("什么", "哪", "吗", "呢", "几", "谁", "怎么",
          "如何", "?", "？")
    return not any(w in query for w in _Q)


class SessionMemory:
    """SQLite 会话记忆 + 事实库。线程安全（写加锁）。"""

    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    # ---------- 底层 ----------
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    def _init_db(self):
        conn = self._conn()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id, ts);

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 0.6,
            created_at INTEGER NOT NULL,
            source TEXT DEFAULT 'rule'
        );
        CREATE INDEX IF NOT EXISTS idx_user_facts ON facts(user_id, created_at DESC);
        """)
        # 兼容旧表：缺列自动补
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(facts)")}
            if "source" not in cols:
                conn.execute("ALTER TABLE facts ADD COLUMN source TEXT DEFAULT 'rule'")
                print("[migrate] facts 加 source 列")
            if "confidence" not in cols:
                conn.execute("ALTER TABLE facts ADD COLUMN confidence REAL DEFAULT 0.6")
                print("[migrate] facts 加 confidence 列")
            if "created_at" not in cols:
                conn.execute("ALTER TABLE facts ADD COLUMN created_at INTEGER DEFAULT 0")
                print("[migrate] facts 加 created_at 列")
            conn.commit()
        except Exception as e:
            print(f"[migrate-warn] {e}")
        conn.close()

    # ---------- 消息 ----------
    def save(self, session_id: str, user_id: str,
             query: str, answer: str):
        """写入一轮对话（user + assistant）。"""
        with self._lock:
            conn = self._conn()
            ts = int(time.time())
            conn.executemany(
                "INSERT INTO messages (session_id,user_id,role,content,ts) "
                "VALUES (?,?,?,?,?)",
                [
                    (session_id, user_id, "user", query, ts),
                    (session_id, user_id, "assistant", answer, ts + 1),
                ],
            )
            conn.commit()
            conn.close()

    def save_async(self, session_id: str, user_id: str,
                   query: str, answer: str):
        """同步保存（<5ms）+ 抽取 facts。"""
        try:
            self.save(session_id, user_id, query, answer)
            n = self.extract_facts(user_id, query)
            print(f"[L1-SAVE] sid={session_id} uid={user_id} "
                  f"facts+={n} q={query[:30]!r}", flush=True)
        except Exception as e:
            import traceback
            print(f"[memory-save-error] {e}", flush=True)
            traceback.print_exc()

    def load_history(self, session_id: str,
                     limit: int = 20) -> List[Dict]:
        """取最近 N 条（时间正序）。"""
        conn = self._conn()
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id = ? ORDER BY ts DESC, id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        conn.close()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    # ---------- 事实 ----------
    def extract_facts(self, user_id: str, query: str):
        """规则抽取，写入 facts 表。"""
        if not user_id or user_id == "anon" or not query:
            return 0
        # 神经网络判断：只从陈述事实的句子抽取
        if not _is_fact_statement(query):
            return 0
        facts = []
        for pat, key in _FACT_RULES:
            for m in re.finditer(pat, query):
                v = m.group(1).strip()
                if 1 < len(v) <= 60:
                    facts.append((key, v))
        if not facts:
            return 0
        added = 0
        with self._lock:
            conn = self._conn()
            now = int(time.time())
            for k, v in facts:
                # 去重：同 user 同 key 同 value 不重复写
                exists = conn.execute(
                    "SELECT 1 FROM facts WHERE user_id=? AND key=? AND value=? LIMIT 1",
                    (user_id, k, v),
                ).fetchone()
                if exists:
                    continue
                conn.execute(
                    "INSERT INTO facts (user_id,key,value,confidence,created_at,source) "
                    "VALUES (?,?,?,?,?,?)",
                    (user_id, k, v, 0.6, now, "rule"),
                )
                added += 1
            conn.commit()
            conn.close()
        return added

    def query_facts(self, user_id: str, query: str = "",
                    top_k: int = 5) -> List[Dict]:
        """查用户事实。有 query 时按关键词命中排序，否则取最近。"""
        if not user_id or user_id == "anon":
            return []
        conn = self._conn()
        rows = conn.execute(
            "SELECT key, value, confidence, created_at FROM facts "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, top_k * 4),
        ).fetchall()
        conn.close()

        items = [{"key": k, "value": v, "conf": c, "ts": t}
                 for k, v, c, t in rows]

        # 简单关键词加权
        if query:
            for it in items:
                score = it["conf"]
                if it["value"] and it["value"] in query:
                    score += 1.0
                elif any(ch in query for ch in it["value"][:3] if ch):
                    score += 0.3
                it["_score"] = score
            items.sort(key=lambda x: -x["_score"])

        return items[:top_k]

    def build_user_context(self, user_id: str, query: str) -> str:
        """给 system 注入的用户信息。无则返回空字符串。"""
        if not user_id or user_id == "anon":
            return ""
        # 只在个人化问题上触发，避免污染所有回复
        if query and not any(t in query for t in _PERSONAL_TRIGGERS):
            return ""
        facts = self.query_facts(user_id, query, top_k=5)
        if not facts:
            return ""
        lines = [f"- {f['key']}: {f['value']}" for f in facts]
        return (
            "【用户之前告诉过你的信息】\n"
            + "\n".join(lines)
            + "\n\n当用户询问自己的信息时（如'我...什么'），"
            "直接引用上面内容回答，不要说不记得或反问。"
        )

    # ---------- 维护 ----------
    def cleanup(self, days: int = 30):
        """清理 N 天前的消息。"""
        cutoff = int(time.time()) - days * 86400
        with self._lock:
            conn = self._conn()
            n = conn.execute(
                "DELETE FROM messages WHERE ts < ?", (cutoff,)
            ).rowcount
            conn.commit()
            conn.close()
        return n


# 单例
_memory: Optional[SessionMemory] = None


def get_memory() -> SessionMemory:
    global _memory
    if _memory is None:
        _memory = SessionMemory()
    return _memory
