"""L3 反馈收集：SQLite 存储 + 候选池导出。"""
import os
import json
import sqlite3
import threading
import time
from typing import List, Dict, Optional

_LOCK = threading.Lock()


class FeedbackStore:
    def __init__(self, db_path: str = "data/feedback.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _conn(self):
        c = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _init_db(self):
        conn = self._conn()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_id TEXT,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            intent TEXT,
            route TEXT,
            adapter TEXT,
            feedback_type TEXT NOT NULL,
            correction TEXT,
            confidence REAL DEFAULT 0.5,
            created_at INTEGER NOT NULL,
            used BOOLEAN DEFAULT 0,
            meta TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_user ON feedback(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_unused ON feedback(used, confidence DESC);
        """)
        conn.commit()
        conn.close()

    def add(self, query: str, answer: str, feedback_type: str,
            session_id: str = None, user_id: str = None,
            intent: str = None, route: str = None, adapter: str = None,
            correction: str = None, confidence: float = 0.5,
            meta: dict = None) -> int:
        """写入反馈。返回 id。"""
        if feedback_type not in ("up", "down", "correct", "followup"):
            raise ValueError(f"未知反馈类型: {feedback_type}")
        # 纠正场景必须有 correction
        if feedback_type == "correct" and not correction:
            raise ValueError("纠正反馈必须提供 correction")

        with _LOCK:
            conn = self._conn()
            cur = conn.execute(
                """INSERT INTO feedback
                (session_id, user_id, query, answer, intent, route, adapter,
                 feedback_type, correction, confidence, created_at, meta)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (session_id, user_id, query, answer, intent, route, adapter,
                 feedback_type, correction, confidence, int(time.time()),
                 json.dumps(meta or {}, ensure_ascii=False)),
            )
            rid = cur.lastrowid
            conn.commit()
            conn.close()
            return rid

    def stats(self) -> Dict:
        conn = self._conn()
        rows = conn.execute(
            """SELECT feedback_type, count(*) FROM feedback
            GROUP BY feedback_type"""
        ).fetchall()
        unused = conn.execute(
            "SELECT count(*) FROM feedback WHERE used=0"
        ).fetchone()[0]
        total = conn.execute("SELECT count(*) FROM feedback").fetchone()[0]
        conn.close()
        return {
            "total": total,
            "unused": unused,
            "by_type": dict(rows),
        }

    def export_candidates(self, min_confidence: float = 0.6,
                          limit: int = 500) -> List[Dict]:
        """导出高置信度未用样本为候选。"""
        conn = self._conn()
        rows = conn.execute(
            """SELECT id, query, answer, correction, feedback_type,
                      confidence, route, adapter
            FROM feedback
            WHERE used=0 AND confidence >= ?
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?""",
            (min_confidence, limit),
        ).fetchall()
        conn.close()

        candidates = []
        for (fid, q, a, corr, ftype, conf, route, adapter) in rows:
            if ftype == "up":
                candidates.append({
                    "source_id": fid,
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                    "type": "positive",
                    "confidence": conf,
                })
            elif ftype == "correct":
                candidates.append({
                    "source_id": fid,
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": corr},
                    ],
                    "type": "correction",
                    "confidence": conf,
                })
            elif ftype == "down":
                # 负样本：暂时不用（可能污染）
                pass
        return candidates

    def mark_used(self, ids: List[int]):
        if not ids:
            return
        with _LOCK:
            conn = self._conn()
            conn.executemany(
                "UPDATE feedback SET used=1 WHERE id=?",
                [(i,) for i in ids],
            )
            conn.commit()
            conn.close()


_singleton = None


def get_feedback_store() -> FeedbackStore:
    global _singleton
    if _singleton is None:
        _singleton = FeedbackStore()
    return _singleton
