"""知识库管理 API。"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from ._db import read_conn
import os, json, sqlite3

router = APIRouter()


def _get_router():
    from serve.app import _state
    r = _state.get("router")
    if r is None:
        raise HTTPException(503, "模型未加载")
    return r


@router.get("/stats")
def kb_stats():
    """KB 统计。"""
    r = _get_router()
    passages = getattr(r, "kb_passages", [])
    return {
        "total": len(passages),
        "emb_dim": r.kb_embs.shape[1] if hasattr(r, "kb_embs") else 0,
        "source": "DuReader",
    }


@router.get("/list")
def kb_list(offset: int = 0, limit: int = 20, q: Optional[str] = None):
    """分页列出 KB 条目（可选关键词过滤）。"""
    r = _get_router()
    passages = getattr(r, "kb_passages", [])
    if q:
        ql = q.lower()
        idx = [i for i, p in enumerate(passages) if ql in p.lower()]
    else:
        idx = list(range(len(passages)))
    page = idx[offset:offset + limit]
    return {
        "total": len(idx),
        "offset": offset,
        "limit": limit,
        "items": [{"id": i, "text": passages[i][:500]} for i in page],
    }


@router.get("/item/{idx}")
def kb_item(idx: int):
    """获取单条。"""
    r = _get_router()
    passages = getattr(r, "kb_passages", [])
    if idx < 0 or idx >= len(passages):
        raise HTTPException(404, "不存在")
    return {"id": idx, "text": passages[idx]}
