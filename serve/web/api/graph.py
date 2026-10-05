"""四维关联图谱（无 emoji，纯颜色）。"""
import os
import time
import datetime
from collections import defaultdict, Counter
from fastapi import APIRouter, HTTPException
from typing import Optional

from ._db import read_conn

router = APIRouter()


# ============ 域 → 颜色（无 emoji） ============
DOMAIN_MAP = {
    "偏好":   ("preference", "#f59e0b"),
    "反偏好": ("preference", "#ef4444"),
    "身份":   ("identity",   "#8b5cf6"),
    "姓名":   ("identity",   "#a855f7"),
    "位置":   ("location",   "#06b6d4"),
    "习惯":   ("habit",      "#10b981"),
    "显式记忆": ("note",     "#3b82f6"),
}
DEFAULT_DOMAIN = ("note", "#64748b")

SOURCE_MAP = {
    "explicit": ("note", "#3b82f6"),
    "implicit": ("note", "#6366f1"),
    "chat":     ("note", "#8b5cf6"),
}


def _classify_fact(key: str):
    for kw, meta in DOMAIN_MAP.items():
        if kw in (key or ""):
            return meta
    return DEFAULT_DOMAIN


def _classify_source(source: str):
    return SOURCE_MAP.get(source or "", DEFAULT_DOMAIN)


def _make_tooltip(category, label, meta):
    lines = [f"<b>{label[:60]}</b>", f"类型: {category}"]
    if meta.get("user_id"): lines.append(f"用户: {meta['user_id']}")
    if meta.get("source"): lines.append(f"来源: {meta['source']}")
    if meta.get("confidence"):
        lines.append(f"置信度: {meta['confidence']:.2f}")
    if meta.get("created_at"):
        ts = meta["created_at"]
        if ts > 1e9:
            dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            lines.append(f"时间: {dt}")
    if meta.get("msg_count"): lines.append(f"消息数: {meta['msg_count']}")
    return "<br>".join(lines)


@router.get("/overview")
def graph_overview(scope: str = "all",
                   user_id: Optional[str] = None,
                   max_nodes: int = 200):
    nodes = []
    edges = []
    node_ids = set()
    node_meta = {}

    def add_node(nid, label, category, domain, color, size=20, **meta):
        if nid in node_ids:
            return
        node_ids.add(nid)
        nodes.append({
            "id": nid,
            "label": label[:40],
            "category": category,
            "domain": domain,
            "color": color,
            "size": size,
            "title": _make_tooltip(category, label, meta),
            **meta,
        })
        node_meta[nid] = {
            "domain": domain,
            "user_id": meta.get("user_id"),
            "ts": meta.get("created_at", 0),
        }

    def add_edge(a, b, etype, score=None):
        if a == b or a not in node_ids or b not in node_ids:
            return
        edges.append({"from": a, "to": b, "type": etype, "score": score})

    # 中心
    if user_id:
        center_id = f"center_{user_id}"
        add_node(center_id, user_id, "center", "center",
                 "#ef4444", size=45, user_id=user_id, is_center=True)
    else:
        center_id = "center_hb"
        add_node(center_id, "HybridBrain", "center", "center",
                 "#ef4444", size=50, is_center=True)

    # L1 facts
    if scope in ("all", "l1") and os.path.exists("data/memory.db"):
        try:
            conn = read_conn("data/memory.db")
            if user_id:
                rows = conn.execute(
                    "SELECT id, user_id, key, value, confidence, "
                    "created_at, source FROM facts WHERE user_id=? LIMIT ?",
                    (user_id, max_nodes)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, user_id, key, value, confidence, "
                    "created_at, source FROM facts LIMIT ?",
                    (max_nodes,)).fetchall()
            conn.close()
        except Exception:
            rows = []

        for r in rows:
            fid = f"fact_{r['id']}"
            domain, color = _classify_fact(r["key"])
            conf = float(r["confidence"] or 0.6)
            size = int(15 + conf * 25)
            add_node(fid, f"{r['key']}:{r['value']}",
                     "fact", domain, color, size=size,
                     user_id=r["user_id"], key=r["key"],
                     value=r["value"], confidence=conf,
                     created_at=r["created_at"] or 0,
                     source=r["source"])
            if r["user_id"] == user_id or not user_id:
                add_edge(center_id, fid, "information")

    # L2 user_kb
    if scope in ("all", "l2") and os.path.exists("data/user_kb.db"):
        try:
            conn = read_conn("data/user_kb.db")
            if user_id:
                rows = conn.execute(
                    "SELECT id, user_id, text, source, created_at "
                    "FROM user_kb WHERE user_id=? LIMIT ?",
                    (user_id, max_nodes)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, user_id, text, source, created_at "
                    "FROM user_kb LIMIT ?", (max_nodes,)).fetchall()
            conn.close()
        except Exception:
            rows = []

        for r in rows:
            nid = f"ukb_{r['id']}"
            domain, color = _classify_source(r["source"])
            size = int(15 + min(len(r["text"]) / 10, 25))
            add_node(nid, r["text"][:50], "user_kb",
                     domain, color, size=size,
                     user_id=r["user_id"], source=r["source"],
                     created_at=r["created_at"] or 0)
            if r["user_id"] == user_id or not user_id:
                add_edge(center_id, nid, "information")

    # sessions
    if scope in ("all", "session") and os.path.exists("data/memory.db"):
        try:
            conn = read_conn("data/memory.db")
            if user_id:
                rows = conn.execute(
                    "SELECT session_id, user_id, count(*) as n, "
                    "max(ts) as last_ts FROM messages WHERE user_id=? "
                    "GROUP BY session_id ORDER BY last_ts DESC LIMIT 30",
                    (user_id,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT session_id, user_id, count(*) as n, "
                    "max(ts) as last_ts FROM messages "
                    "GROUP BY session_id ORDER BY last_ts DESC LIMIT 30"
                ).fetchall()
            conn.close()
        except Exception:
            rows = []

        for r in rows:
            sid = f"sess_{r['session_id']}"
            size = min(15 + r["n"], 40)
            add_node(sid, f"{r['session_id'][:10]}({r['n']})",
                     "session", "session", "#64748b",
                     size=size, user_id=r["user_id"],
                     created_at=r["last_ts"] or 0, msg_count=r["n"])
            add_edge(center_id, sid, "information")

    # 内部关联
    _build_internal_edges(nodes, edges)

    by_domain = Counter(n["domain"] for n in nodes)
    by_type = Counter(n["category"] for n in nodes)
    edge_types = Counter(e["type"] for e in edges)

    return {
        "nodes": nodes,
        "edges": edges,
        "legend": {
            "domains": {
                "preference": {"label": "偏好", "color": "#f59e0b"},
                "identity":   {"label": "身份", "color": "#8b5cf6"},
                "location":   {"label": "位置", "color": "#06b6d4"},
                "habit":      {"label": "习惯", "color": "#10b981"},
                "note":       {"label": "笔记", "color": "#3b82f6"},
                "session":    {"label": "会话", "color": "#64748b"},
                "center":     {"label": "中心", "color": "#ef4444"},
            },
            "edges": {
                "semantic":    {"color": "#22c55e", "label": "语义相似"},
                "knowledge":   {"color": "#3b82f6", "label": "同域知识"},
                "information": {"color": "#f97316", "label": "同源信息"},
            }
        },
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "by_domain": dict(by_domain),
            "by_type": dict(by_type),
            "edge_types": dict(edge_types),
        }
    }


def _build_internal_edges(nodes, edges):
    by_user = defaultdict(list)
    by_domain = defaultdict(list)
    by_key = defaultdict(list)

    for n in nodes:
        nid = n["id"]
        if n.get("is_center"):
            continue
        if n.get("user_id"):
            by_user[n["user_id"]].append(nid)
        if n.get("category") == "fact" and n.get("key"):
            by_key[n["key"]].append(nid)
        if n.get("domain"):
            by_domain[n["domain"]].append(nid)

    for key, ids in by_key.items():
        for i in range(len(ids) - 1):
            edges.append({"from": ids[i], "to": ids[i+1],
                          "type": "knowledge"})

    for domain, ids in by_domain.items():
        if len(ids) < 2 or domain in ("center", "session"):
            continue
        for i, nid in enumerate(ids):
            for j in range(i + 1, min(i + 3, len(ids))):
                edges.append({"from": nid, "to": ids[j],
                              "type": "knowledge"})

    for uid, ids in by_user.items():
        if len(ids) < 2:
            continue
        for i in range(len(ids) - 1):
            a, b = ids[i], ids[i+1]
            if not any((e["from"] == a and e["to"] == b)
                       or (e["from"] == b and e["to"] == a)
                       for e in edges[-30:]):
                edges.append({"from": a, "to": b,
                              "type": "information"})
