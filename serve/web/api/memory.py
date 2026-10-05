"""记忆库管理 API（L1 facts + L2 user_kb + session）。"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from ._db import read_conn
import os, sqlite3, pickle

router = APIRouter()


@router.get("/l1/users")
def l1_users():
    """L1 facts 表：按用户聚合。"""
    path = "data/memory.db"
    if not os.path.exists(path):
        return {"users": []}
    conn = read_conn(path)
    rows = conn.execute(
        "SELECT user_id, count(*) FROM facts GROUP BY user_id"
    ).fetchall()
    conn.close()
    return {"users": [{"user_id": u, "count": c} for u, c in rows]}


@router.get("/l1/facts")
def l1_facts(user_id: Optional[str] = None, limit: int = 50):
    path = "data/memory.db"
    if not os.path.exists(path):
        return {"items": []}
    conn = read_conn(path)
    if user_id:
        rows = conn.execute(
            "SELECT * FROM facts WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM facts ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows]}


@router.get("/l1/sessions")
def l1_sessions(limit: int = 30):
    """会话列表。"""
    path = "data/memory.db"
    if not os.path.exists(path):
        return {"items": []}
    conn = read_conn(path)
    rows = conn.execute(
        """SELECT session_id, user_id, count(*) as n,
                  max(ts) as last_ts
           FROM messages GROUP BY session_id
           ORDER BY last_ts DESC LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()
    return {"items": [{"session_id": s, "user_id": u, "n": n, "last_ts": t}
                      for s, u, n, t in rows]}


@router.get("/l1/session/{session_id}")
def l1_session_detail(session_id: str, limit: int = 100):
    path = "data/memory.db"
    if not os.path.exists(path):
        return {"items": []}
    conn = read_conn(path)
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id=? ORDER BY ts ASC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows]}


@router.get("/l2/users")
def l2_users():
    path = "data/user_kb.db"
    if not os.path.exists(path):
        return {"users": []}
    conn = read_conn(path)
    rows = conn.execute(
        "SELECT user_id, count(*) FROM user_kb GROUP BY user_id"
    ).fetchall()
    conn.close()
    return {"users": [{"user_id": u, "count": c} for u, c in rows]}


@router.get("/l2/items")
def l2_items(user_id: Optional[str] = None, limit: int = 30):
    path = "data/user_kb.db"
    if not os.path.exists(path):
        return {"items": []}
    conn = read_conn(path)
    if user_id:
        rows = conn.execute(
            "SELECT id, user_id, text, source, created_at FROM user_kb "
            "WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, user_id, text, source, created_at FROM user_kb "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows]}
