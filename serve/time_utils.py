"""时间归一化工具。"""
from datetime import datetime, timezone, timedelta
from typing import Optional, Union


CST = timezone(timedelta(hours=8))


def parse_timestamp(ts: Optional[Union[str, int, float]]
                    ) -> Optional[datetime]:
    """把各种格式归一化为带时区的 datetime。

    支持：
      - ISO 8601 字符串
      - Unix 毫秒或秒
      - None → 返回 None
    """
    if ts is None:
        return None

    if isinstance(ts, (int, float)):
        if ts > 1e10:
            ts = ts / 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=CST)
        except (ValueError, OSError):
            return None

    if isinstance(ts, str):
        ts = ts.strip()
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=CST)
            return dt
        except ValueError:
            return None

    return None


def time_ago(dt: Optional[datetime]) -> Optional[str]:
    """人类可读的"多久以前"。"""
    if dt is None:
        return None
    now = datetime.now(CST)
    sec = (now - dt).total_seconds()

    if sec < 0:
        return None  # 未来时间，忽略
    if sec < 60:
        return "刚刚"
    if sec < 3600:
        return f"{int(sec // 60)} 分钟前"
    if sec < 86400:
        return f"{int(sec // 3600)} 小时前"
    if sec < 86400 * 7:
        return f"{int(sec // 86400)} 天前"
    if sec < 86400 * 30:
        return f"{int(sec // (86400 * 7))} 周前"
    if sec < 86400 * 365:
        return f"{int(sec // (86400 * 30))} 个月前"
    return f"{int(sec // (86400 * 365))} 年前"


def format_time_context(dt: Optional[datetime]) -> str:
    """给 prompt 用的时间描述。"""
    if dt is None:
        return ""
    now = datetime.now(CST)

    # 未来时间 → 忽略
    if (dt - now).total_seconds() > 60:
        return ""
    # 太旧 → 忽略
    if (now - dt).days > 365:
        return ""

    if dt.date() == now.date():
        return f"今天 {dt.strftime('%H:%M')}"
    if (now.date() - dt.date()).days == 1:
        return f"昨天 {dt.strftime('%H:%M')}"

    ago = time_ago(dt)
    return f"{dt.strftime('%Y-%m-%d %H:%M')}（{ago}）"


def history_span(history: list) -> str:
    """历史消息的时间跨度。"""
    if not history:
        return ""
    timestamps = [parse_timestamp(h.get("timestamp")) for h in history]
    timestamps = [t for t in timestamps if t is not None]
    if len(timestamps) < 2:
        return ""
    span = max(timestamps) - min(timestamps)
    sec = span.total_seconds()
    if sec < 3600:
        return f"{int(sec // 60)} 分钟"
    if sec < 86400:
        return f"{int(sec // 3600)} 小时"
    return f"{int(sec // 86400)} 天"
