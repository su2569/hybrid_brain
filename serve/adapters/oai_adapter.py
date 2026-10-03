"""OpenAI 兼容协议适配。"""
import time
import uuid

from serve.adapters.utils import parse_oai_content
from serve.adapters.validators import (
    sanitize_nickname, sanitize_user_id, normalize_persona,
)


def parse(req) -> dict:
    """OpenAI 请求 → 内部标准格式。"""
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise ValueError("至少需要一条 user message")
    last = user_msgs[-1]

    # 解析最后一条 user 消息
    parsed = parse_oai_content(last.content)

    # 历史
    history = []
    for m in req.messages[:-1]:
        if m.role == "system":
            continue
        if m.role == "tool":
            history.append({
                "role": "tool",
                "content": str(m.content or ""),
                "tool_call_id": m.tool_call_id,
            })
            continue
        item = {"role": m.role, "content": str(m.content or "")}
        if m.hybrid_brain_ts is not None:
            item["timestamp"] = m.hybrid_brain_ts
        history.append(item)

    # 用户身份（OpenAI 原生 user 字段是 id）
    oai_user = req.hybrid_brain_user or {}
    user = {
        "id": sanitize_user_id(req.user),
        "nickname": sanitize_nickname(oai_user.get("nickname")),
        "secondary_nickname": sanitize_nickname(
            oai_user.get("secondary_nickname")),
    }

    # 控制字段
    ctrl = req.hybrid_brain_control or {}
    control = {
        "persona": normalize_persona(ctrl.get("persona", "cyrene")),
        "enable_thinking": ctrl.get("enable_thinking"),
        "show_thinking": bool(ctrl.get("show_thinking", False)),
        "thinking_budget": int(ctrl.get("thinking_budget", 2000)),
    }

    return {
        "query": parsed["text"],
        "attachments": parsed["attachments"],
        "history": history,
        "query_ts": last.hybrid_brain_ts,
        "user": user,
        "control": control,
        "tools": req.tools or [],
    }


def format_response(result: dict, req, elapsed_ms: int) -> dict:
    """内部结果 → OpenAI 兼容响应。"""
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    message = {
        "role": "assistant",
        "content": result.get("answer", ""),
    }

    # 工具调用
    tool_calls = result.get("tool_calls")
    if tool_calls:
        message["content"] = None
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop"

    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(result.get("answer", "")),
            "total_tokens": len(result.get("answer", "")),
        },
        # 扩展位：标准客户端忽略
        "hybrid_brain": {
            "intent": result.get("intent", "chat"),
            "intent_conf": float(result.get("intent_conf", 0.0)),
            "persona": result.get("persona", "cyrene"),
            "sources": (result.get("sources") or [])[:3],
            "latency_ms": elapsed_ms,
        },
    }


def format_stream_chunk(text: str, model: str, resp_id: str,
                        is_first: bool = False,
                        is_last: bool = False) -> str:
    """构造 SSE chunk（返回字符串，由 app.py 输出）。"""
    import json

    if is_first:
        delta = {"role": "assistant"}
    elif is_last:
        delta = {}
    else:
        delta = {"content": text}

    chunk = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": "stop" if is_last else None,
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
