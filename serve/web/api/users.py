"""用户管理：user_id ↔ 显示名/系统编号。"""
import os
import json
import time
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException, Body
from typing import Optional

router = APIRouter()

USERS_FILE = Path("data/users.json")


def _load() -> dict:
    if not USERS_FILE.exists():
        return {"users": {}}
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"users": {}}


def _save(data: dict):
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = USERS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(USERS_FILE)


def _gen_sys_id() -> str:
    """系统编号：SYS-YYYYMMDD-XXXX"""
    ts = time.strftime("%Y%m%d")
    return f"SYS-{ts}-{uuid.uuid4().hex[:4].upper()}"


@router.get("/list")
def list_users():
    d = _load()
    out = []
    for uid, info in d.get("users", {}).items():
        out.append({
            "user_id": uid,
            "display_name": info.get("display_name", uid),
            "system_id": info.get("system_id", ""),
            "created_at": info.get("created_at", 0),
            "note": info.get("note", ""),
        })
    out.sort(key=lambda x: -x["created_at"])
    return {"users": out}


@router.post("/create")
def create_user(body: dict = Body(...)):
    """创建用户映射。
    body: {"user_id": "...", "display_name": "...", "system_id": "...(可选)"}
    """
    user_id = (body.get("user_id") or "").strip()
    if not user_id:
        raise HTTPException(400, "user_id 必填")
    if not user_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "user_id 只允许字母数字-_")

    d = _load()
    users = d.setdefault("users", {})
    if user_id in users:
        raise HTTPException(400, f"已存在: {user_id}")

    users[user_id] = {
        "display_name": body.get("display_name") or user_id,
        "system_id": body.get("system_id") or _gen_sys_id(),
        "note": body.get("note", ""),
        "created_at": int(time.time()),
    }
    _save(d)
    return {"ok": True, "user": users[user_id]}


@router.put("/update")
def update_user(body: dict = Body(...)):
    user_id = (body.get("user_id") or "").strip()
    if not user_id:
        raise HTTPException(400, "user_id 必填")
    d = _load()
    users = d.setdefault("users", {})
    if user_id not in users:
        raise HTTPException(404, f"不存在: {user_id}")

    for k in ("display_name", "system_id", "note"):
        if k in body:
            users[user_id][k] = body[k]
    _save(d)
    return {"ok": True, "user": users[user_id]}


@router.delete("/delete")
def delete_user(user_id: str):
    d = _load()
    users = d.setdefault("users", {})
    if user_id in users:
        del users[user_id]
        _save(d)
    return {"ok": True}


@router.get("/lookup/{user_id}")
def lookup(user_id: str):
    d = _load()
    u = d.get("users", {}).get(user_id)
    if not u:
        # 自动创建
        users = d.setdefault("users", {})
        users[user_id] = {
            "display_name": user_id,
            "system_id": _gen_sys_id(),
            "note": "",
            "created_at": int(time.time()),
        }
        _save(d)
        u = users[user_id]
    return {"user_id": user_id, **u}


@router.post("/auto_register")
def auto_register(body: dict = Body(...)):
    """根据 memory.db 中实际用户，自动注册未登记的。"""
    from serve.memory import db_manager as dm
    d = _load()
    users = d.setdefault("users", {})
    added = 0
    # L1 facts
    try:
        rows = dm.query("memory",
                        "SELECT DISTINCT user_id FROM facts")
        for r in rows:
            uid = r["user_id"]
            if uid and uid not in users and uid != "anon":
                users[uid] = {
                    "display_name": uid,
                    "system_id": _gen_sys_id(),
                    "note": "自动注册",
                    "created_at": int(time.time()),
                }
                added += 1
    except Exception:
        pass
    # L2 user_kb
    try:
        rows = dm.query("user_kb",
                        "SELECT DISTINCT user_id FROM user_kb")
        for r in rows:
            uid = r["user_id"]
            if uid and uid not in users and uid != "anon":
                users[uid] = {
                    "display_name": uid,
                    "system_id": _gen_sys_id(),
                    "note": "自动注册",
                    "created_at": int(time.time()),
                }
                added += 1
    except Exception:
        pass
    _save(d)
    return {"ok": True, "added": added}
