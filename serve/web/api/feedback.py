"""反馈管理 API。"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from serve.memory import db_manager as dm

router = APIRouter()


@router.get("/list")
def list_feedback(feedback_type: Optional[str] = None,
                  used: Optional[int] = None,
                  limit: int = 50, offset: int = 0):
    where = []
    params = []
    if feedback_type:
        where.append("feedback_type=?")
        params.append(feedback_type)
    if used is not None:
        where.append("used=?")
        params.append(used)
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""

    with dm.connect("feedback", readonly=True) as conn:
        total = conn.execute(
            f"SELECT count(*) FROM feedback{where_clause}",
            tuple(params)
        ).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM feedback{where_clause} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
            tuple(params) + (limit, offset)
        ).fetchall()
        items = [dict(r) for r in rows]

    # 统计
    by_type = {}
    with dm.connect("feedback", readonly=True) as conn:
        for r in conn.execute(
            "SELECT feedback_type, count(*) FROM feedback "
            "GROUP BY feedback_type"
        ).fetchall():
            by_type[r[0]] = r[1]
        unused = conn.execute(
            "SELECT count(*) FROM feedback WHERE used=0"
        ).fetchone()[0]

    return {
        "total": total, "offset": offset, "limit": limit,
        "items": items,
        "stats": {"by_type": by_type, "unused": unused},
    }


@router.post("/export")
def export_candidates(min_confidence: float = 0.6, limit: int = 500):
    from serve.memory.feedback import get_feedback_store
    store = get_feedback_store()
    cands = store.export_candidates(min_confidence, limit)
    return {"count": len(cands), "candidates": cands}


@router.post("/mark_used")
def mark_used(ids: list):
    from serve.memory.feedback import get_feedback_store
    store = get_feedback_store()
    store.mark_used(ids)
    return {"ok": True, "n": len(ids)}


@router.delete("/delete")
def delete_feedback(fid: int):
    dm.execute("feedback", "DELETE FROM feedback WHERE id=?", (fid,))
    return {"ok": True}
