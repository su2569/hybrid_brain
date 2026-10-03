"""请求/响应模型。"""
from typing import List, Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    history: List[Message] = []
    top_k: int = 5


class Source(BaseModel):
    passage: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    intent: str
    intent_conf: float
    sources: List[Source] = []


class HealthResponse(BaseModel):
    status: str
    kb_size: int
