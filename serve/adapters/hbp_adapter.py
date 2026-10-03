"""HBP/1.0 协议适配。"""
import time
import uuid

from serve.adapters.validators import (
    sanitize_nickname, sanitize_user_id, normalize_persona,
)


def parse(req) -> dict:
    """HBP 请求 → 内部标准格式。"""
    # 用户身份清洗
    user = {
        "id": sanitize_user_id(req.user.id),
        "nickname": sanitize_nickname(req.user.nickname),
        "secondary_nickname": sanitize_nickname(req.user.secondary_nickname),
    }

    # 控制字段
    control = {
        "persona": normalize_persona(req.control.persona),
        "enable_thinking": req.control.enable_thinking,
        "show_thinking": req.control.show_thinking,
        "thinking_budget": req.control.thinking_budget,
        "allow_proactive": getattr(req.control, "allow_proactive", False),
    }

    return {
        "query": req.query.text or "",
        "attachments": [a.model_dump() for a in req.query.attachments],
        "history": [h.model_dump() for h in req.query.history],
        "query_ts": req.query.timestamp,
        "user": user,
        "control": control,
        "tools": req.tools or [],
    }


def format_response(result: dict, req, elapsed_ms: int) -> dict:
    """内部结果 → HBP 完整响应。"""
    resp_id = f"resp_{uuid.uuid4().hex[:12]}"

    answer = {
        "text": result.get("answer", ""),
        "persona": result.get("persona", "cyrene"),
        "finalized": result.get("finalized", False),
    }
    # 主动调用（始终写入，哪怕空列表）
    proactive = result.get("proactive_calls") or []
    answer["proactive_calls"] = proactive
    if result.get("thinking"):
        answer["thinking"] = result["thinking"]

    # 来源
    sources = result.get("sources") or []
    sources_out = [
        {
            "passage": s.get("passage", "")[:500],
            "score": float(s.get("score", 0.0)),
        }
        for s in sources[:5]
    ]

    return {
        "hbp_version": "1.0",
        "response_id": resp_id,
        "answer": answer,
        "routing": {
            "intent": result.get("intent", "chat"),
            "confidence": float(result.get("intent_conf", 0.0)),
            "path": result.get("path", result.get("intent", "chat")),
        },
        "retrieval": {
            "sources": sources_out,
        },
        "memory": result.get("memory", {}),
        "finalize": result.get("finalize", {}),
        "control": {
            "persona": result.get("persona", "cyrene"),
            "enable_thinking": result.get("enable_thinking", True),
            "show_thinking": bool(result.get("thinking")),
        },
        "time": {
            "elapsed_ms": elapsed_ms,
        },
        "user": {
            "id": (req.user.id or "anonymous"),
            "display_name": result.get("display_name"),
        },
    }
