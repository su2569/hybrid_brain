"""HybridBrain FastAPI 服务（双协议）。"""
import os
import sys
import time
import json
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from serve.schemas import (
    HBPRequest, OAIRequest,
    ChatRequest, ChatResponse, Source, HealthResponse,
)
from serve.adapters import hbp_adapter, oai_adapter


# ============================================================
# 全局单例
# ============================================================
_state = {"router": None, "load_time": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动自检
    try:
        from serve.preflight import check_all
        check_all(strict=False, verbose=True)
    except Exception as e:
        print(f"[preflight] 跳过: {e}")

    print("[startup] 加载 HybridBrainRouter...")
    t0 = time.time()
    from serve.router import HybridBrainRouter
    _state["router"] = HybridBrainRouter()
    _state["load_time"] = time.time() - t0
    print(f"[startup] 加载完成，耗时 {_state['load_time']:.1f}s")
    yield
    print("[shutdown] 清理...")
    _state["router"] = None


app = FastAPI(title="HybridBrain", version="1.2.0", lifespan=lifespan)


# ============================================================
# 通用
# ============================================================
@app.get("/health", response_model=HealthResponse)
def health():
    r = _state["router"]
    if r is None:
        return HealthResponse(status="loading", kb_size=0,
                              model_loaded=False)
    return HealthResponse(status="ok",
                          kb_size=len(getattr(r, "kb_passages", [])),
                          model_loaded=True)


@app.get("/v1/models")
def list_models():
    """OpenAI 兼容的模型列表。"""
    return {
        "object": "list",
        "data": [{
            "id": "hybrid-brain",
            "object": "model",
            "created": 1791002543,
            "owned_by": "su2569",
            "hybrid_brain_capabilities": {
                "modalities": ["text"],
                "features": ["tool_calls", "streaming", "thinking"],
                "personas": ["cyrene", "off"],
            },
        }],
    }


@app.get("/v1/hbp/models")
def hbp_models():
    """HBP 模型列表。"""
    return {
        "hbp_version": "1.0",
        "models": [{
            "id": "hybrid-brain",
            "name": "HybridBrain",
            "version": "1.2.0",
            "modalities": {
                "text": {"enabled": True, "max_length": 2000},
                "image": {"enabled": False},
                "audio": {"enabled": False},
                "video": {"enabled": False},
            },
            "features": {
                "tool_calls": True,
                "streaming": True,
                "thinking": True,
                "persona": True,
                "history": True,
                "timestamp": True,
            },
            "personas": ["cyrene", "off"],
            "context_length": 40960,
        }],
    }


@app.get("/v1/hbp/schema")
def hbp_schema():
    """协议自描述。"""
    return {
        "hbp_version": "1.0",
        "endpoints": {
            "chat": "/v1/hbp/chat",
            "stream": "/v1/hbp/chat/stream",
            "models": "/v1/hbp/models",
        },
        "capabilities": {
            "modalities": ["text"],
            "features": ["tool_calls", "streaming", "thinking"],
            "personas": ["cyrene", "off"],
        },
    }


# ============================================================
# 核心对话
# ============================================================
def _call_router(internal: dict, verbose=True):
    """统一调用 router。"""
    r = _state["router"]
    if r is None:
        raise HTTPException(503, "模型未加载")

    # router.route 签名可能带 control 也可能不带
    try:
        return r.route(
            internal["query"],
            history=internal.get("history"),
            control=internal.get("control"),
            user=internal.get("user"),
            query_ts=internal.get("query_ts"),
            tools=internal.get("tools"),
            extra_system=internal.get("extra_system"),
            session_id=internal.get("session_id"),
        )
    except TypeError as _e:
        # DEBUG: 打印真实错误，不静默兜底
        import traceback
        print(f"[CALL-ROUTER-ERROR] TypeError: {_e}", flush=True)
        traceback.print_exc()
        # 老 router 不支持 control 参数 → 退回
        result = r.route(internal["query"], history=internal.get("history"))
        control = internal.get("control") or {}
        result["persona"] = control.get("persona", "cyrene")
        return result


@app.post("/v1/hbp/chat")
def hbp_chat(req: HBPRequest):
    t0 = time.time()
    internal = hbp_adapter.parse(req)
    result = _call_router(internal)
    elapsed = int((time.time() - t0) * 1000)
    return hbp_adapter.format_response(result, req, elapsed)


@app.post("/v1/chat/completions")
def oai_chat(req: OAIRequest):
    if req.stream:
        return oai_chat_stream(req)

    t0 = time.time()
    try:
        internal = oai_adapter.parse(req)
    except ValueError as e:
        raise HTTPException(400, str(e))

    result = _call_router(internal)
    elapsed = int((time.time() - t0) * 1000)
    return oai_adapter.format_response(result, req, elapsed)


def oai_chat_stream(req: OAIRequest):
    """OpenAI 兼容的流式响应。"""
    import uuid

    try:
        internal = oai_adapter.parse(req)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 先拿到完整答案（当前 router 不支持真流式）
    result = _call_router(internal)
    raw_answer = result.get("answer", "") or ""
    calls = result.get("proactive_calls") or []
    tool_calls = oai_adapter._calls_to_tool_calls(calls)
    import re as _re
    content = oai_adapter._CALL_RE.sub("", raw_answer).strip()
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    def gen():
        # 第一个 chunk：role（含 tool_calls 一起发）
        yield oai_adapter.format_stream_chunk(
            "", req.model, resp_id, is_first=True,
            tool_calls=tool_calls if tool_calls else None)
        # 逐字发 content（如果有）
        if content:
            for ch in content:
                yield oai_adapter.format_stream_chunk(ch, req.model, resp_id)
        # 结束 chunk
        yield oai_adapter.format_stream_chunk(
            "", req.model, resp_id, is_last=True,
            finish_reason="tool_calls" if tool_calls else "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ============================================================
# 旧 /chat 兼容
# ============================================================
@app.post("/chat", response_model=ChatResponse)
def legacy_chat(req: ChatRequest):
    r = _state["router"]
    if r is None:
        raise HTTPException(503, "模型未加载")

    result = r.route(req.query, history=req.history,
                     control=req.control, user=req.user,
                     session_id=req.session_id)
    sources = [
        Source(passage=s["passage"][:500], score=s["score"])
        for s in (result.get("sources") or [])[:3]
    ]
    # /chat 内部私有协议：返回含 <call> 的原始 answer
    answer = result.get("answer_with_calls") or result["answer"]
    return ChatResponse(
        answer=answer,
        intent=result["intent"],
        intent_conf=result["intent_conf"],
        sources=sources,
        session_id=req.session_id,
    )


# ============================================================
# Web UI
# ============================================================
app.mount("/static", StaticFiles(directory="serve/static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    with open("serve/static/index.html", encoding="utf-8") as f:
        return f.read()
