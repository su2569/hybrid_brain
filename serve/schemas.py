"""HybridBrain 双协议 Schema。

包含：
  - 共享类型（UserProfile, Attachment, ControlOptions）
  - HBP 协议（HBPRequest, HBPQuery）
  - OpenAI 兼容协议（OAIRequest, OAIMessage）
  - 响应模型
"""
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field


# ============================================================
# 共享类型
# ============================================================
TimestampType = Optional[Union[str, int, float]]


class UserProfile(BaseModel):
    """客户端发送的用户身份（全可选）。"""
    id: Optional[str] = Field(None, min_length=1, max_length=64)
    nickname: Optional[str] = Field(None, min_length=1, max_length=20)
    secondary_nickname: Optional[str] = Field(None, min_length=1, max_length=20)


class AttachmentSource(BaseModel):
    kind: Literal["url", "base64", "oss"]
    url: Optional[str] = None
    data: Optional[str] = None
    encoding: Optional[str] = None
    bucket: Optional[str] = None
    key: Optional[str] = None
    region: Optional[str] = None
    signed_url: Optional[str] = None
    expires_at: Optional[str] = None


class Attachment(BaseModel):
    type: Literal["image", "audio", "video", "file"]
    mime: str
    source: AttachmentSource
    meta: dict = {}
    id: Optional[str] = None


class ControlOptions(BaseModel):
    """控制开关（全可选，有默认值）。"""
    persona: Literal["cyrene", "off"] = "cyrene"
    enable_thinking: Optional[bool] = None
    show_thinking: bool = False
    thinking_budget: int = Field(2000, ge=100, le=8000)
    use_cyrene_lora: bool = True
    allow_proactive: bool = False


# ============================================================
# HBP 协议
# ============================================================
class HBPMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    attachments: List[Attachment] = []
    timestamp: TimestampType = None


class HBPQuery(BaseModel):
    text: str = Field("", max_length=2000)
    attachments: List[Attachment] = []
    history: List[HBPMessage] = []
    timestamp: TimestampType = None


class HBPRequest(BaseModel):
    hbp_version: str = "1.0"
    query: HBPQuery
    user: UserProfile = UserProfile()
    control: ControlOptions = ControlOptions()
    tools: List[dict] = []


# ============================================================
# OpenAI 兼容协议
# ============================================================
class OAIContentItem(BaseModel):
    type: Literal["text", "image_url", "input_audio", "video_url", "audio_url"]
    text: Optional[str] = None
    image_url: Optional[dict] = None
    input_audio: Optional[dict] = None
    video_url: Optional[dict] = None
    audio_url: Optional[dict] = None


class OAIMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, list]] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    hybrid_brain_ts: TimestampType = None
    hybrid_brain_attachments: Optional[list] = None


class OAIRequest(BaseModel):
    model: str = "hybrid-brain"
    session_id: Optional[str] = Field(None, max_length=64)
    messages: List[OAIMessage]
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None
    user: Optional[str] = None
    hybrid_brain_user: Optional[dict] = None
    hybrid_brain_control: Optional[dict] = None


# ============================================================
# 响应模型
# ============================================================
class Source(BaseModel):
    passage: str
    score: float


class ChatRequest(BaseModel):
    """旧 /chat 端点兼容。"""
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, max_length=64)
    history: List[dict] = []
    user: Optional[dict] = None
    control: Optional[dict] = None
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    intent: str
    intent_conf: float
    sources: List[Source] = []
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    kb_size: int
    model_loaded: bool
