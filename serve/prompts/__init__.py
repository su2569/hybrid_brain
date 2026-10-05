"""统一 prompt 注册。

模式映射：
  chat      诗意昔涟（默认）
  plain     平实昔涟（v9 非诗数据）
  neutral   通用 AI
  tool      工具调用
  fresh     首次问个人信息
  forget    问个人信息但忘了
  defensive 反 abuse
  refuse    拒绝 manipulate
"""
from .chat import CHAT_SYSTEM, PLAIN_SYSTEM, NEUTRAL_SYSTEM
from .rag import RAG_SYSTEM
from .tool import TOOL_SYSTEM
from .personal import PERSONAL_FRESH_SYSTEM, PERSONAL_FORGET_SYSTEM
from .safety import DEFENSIVE_SYSTEM, REFUSE_SYSTEM

PROMPTS = {
    "chat":      CHAT_SYSTEM,
    "plain":     PLAIN_SYSTEM,
    "neutral":   NEUTRAL_SYSTEM,
    "rag":       RAG_SYSTEM,
    "tool":      TOOL_SYSTEM,
    "fresh":     PERSONAL_FRESH_SYSTEM,
    "forget":    PERSONAL_FORGET_SYSTEM,
    "defensive": DEFENSIVE_SYSTEM,
    "refuse":    REFUSE_SYSTEM,
}

PROMPT_NAMES = list(PROMPTS.keys())


def get_prompt(name: str, default: str = "chat") -> str:
    return PROMPTS.get(name, PROMPTS[default])


__all__ = [
    "CHAT_SYSTEM", "PLAIN_SYSTEM", "NEUTRAL_SYSTEM",
    "RAG_SYSTEM", "TOOL_SYSTEM",
    "PERSONAL_FRESH_SYSTEM", "PERSONAL_FORGET_SYSTEM",
    "DEFENSIVE_SYSTEM", "REFUSE_SYSTEM",
    "PROMPTS", "PROMPT_NAMES", "get_prompt",
]
