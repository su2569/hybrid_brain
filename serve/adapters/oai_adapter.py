"""OpenAI 兼容协议适配。"""
import json
import re
import time
import uuid

from serve.adapters.utils import parse_oai_content
from serve.adapters.validators import (
    sanitize_nickname, sanitize_user_id, normalize_persona,
)


def _build_tools_block(tools) -> str:
    """把 OpenAI tools 声明拼成一段 system 文本，注入 TOOL_SYSTEM。"""
    if not tools:
        return ""
    lines = []
    for t in tools:
        f = t.get("function", t) if isinstance(t, dict) else {}
        name = f.get("name", "")
        desc = f.get("description", "")
        props = (f.get("parameters") or {}).get("properties", {}) or {}
        params_str = ", ".join(
            f"{k}: {v.get('type', 'str')}" for k, v in props.items()
        ) if props else ""
        lines.append(f"- {name}({params_str}) — {desc}" if params_str
                     else f"- {name}() — {desc}")
    return (
        "\n\n【本次调用方声明的可用工具】\n"
        + "\n".join(lines)
        + "\n\n【调用规则】"
        + "\n1. 用户问题缺关键参数（型号/城市/时间/表达式等）时，优先用 clarify："
        + "\n   <call type=\"clarify\">请问您说的是哪一款呢？</call>"
        + "\n2. 参数齐全且明确匹配某个工具时，输出 action："
        + "\n   <call type=\"action\" name=\"工具名\" params='{\"参数\":\"值\"}'>简短说明</call>"
        + "\n3. 无关工具时正常对话，不要输出 <call>。"
    )


def parse(req) -> dict:
    """OpenAI 请求 → 内部标准格式。"""
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise ValueError("至少需要一条 user message")
    last = user_msgs[-1]

    parsed = parse_oai_content(last.content)

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

    oai_user = req.hybrid_brain_user or {}
    user = {
        "id": sanitize_user_id(req.user),
        "nickname": sanitize_nickname(oai_user.get("nickname")),
        "secondary_nickname": sanitize_nickname(
            oai_user.get("secondary_nickname")),
    }

    ctrl = req.hybrid_brain_control or {}
    tools = req.tools or []
    # 有 tools 声明 → 自动走 tool 模式
    allow_proactive = bool(ctrl.get("allow_proactive", False)) or bool(tools)
    control = {
        "persona": normalize_persona(ctrl.get("persona", "cyrene")),
        "enable_thinking": ctrl.get("enable_thinking"),
        "show_thinking": bool(ctrl.get("show_thinking", False)),
        "thinking_budget": int(ctrl.get("thinking_budget", 2000)),
        "allow_proactive": allow_proactive,
    }

    return {
        "query": parsed["text"],
        "attachments": parsed["attachments"],
        "history": history,
        "query_ts": last.hybrid_brain_ts,
        "user": user,
        "control": control,
        "tools": tools,
        "extra_system": _build_tools_block(tools),
    }


_CALL_RE = re.compile(
    r'<call[^>]*(?:/>|>.*?</call>)', re.DOTALL)


def _calls_to_tool_calls(calls):
    """proactive_calls → OpenAI tool_calls。仅 action 类型转换。"""
    out = []
    for c in calls or []:
        ctype = c.get("type")
        if ctype != "action":
            continue
        name = c.get("name") or c.get("action")
        if not name:
            continue
        args = c.get("params") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        out.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    return out


def format_response(result: dict, req, elapsed_ms: int) -> dict:
    """内部结果 → OpenAI 兼容响应。"""
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    raw_answer = result.get("answer", "") or ""
    calls = result.get("proactive_calls") or []
    tool_calls = _calls_to_tool_calls(calls)

    # 剥掉 <call> 标签，保留文本（clarify/suggest 也在这里）
    content = _CALL_RE.sub("", raw_answer).strip()
    # 如果只剩空 content 且有 tool_calls，content 设 None
    if not content and tool_calls:
        content = None

    message = {"role": "assistant", "content": content}
    if tool_calls:
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
            "completion_tokens": len(raw_answer),
            "total_tokens": len(raw_answer),
        },
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
                        is_last: bool = False,
                        tool_calls=None,
                        finish_reason=None) -> str:
    """构造 SSE chunk（返回字符串，由 app.py 输出）。"""
    if is_first:
        delta = {"role": "assistant"}
        if tool_calls:
            delta["tool_calls"] = [
                {
                    "index": i,
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }
                for i, tc in enumerate(tool_calls)
            ]
    elif is_last:
        delta = {}
    else:
        delta = {"content": text}

    final_reason = finish_reason
    if is_last and final_reason is None:
        final_reason = "stop"

    chunk = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": final_reason,
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
