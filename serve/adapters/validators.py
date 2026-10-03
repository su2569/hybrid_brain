"""请求字段校验。校验失败静默丢弃，不阻塞主流程。"""
import re
import ipaddress
from typing import Optional
from urllib.parse import urlparse


# 昵称：至少 1 个中文/字母/数字
_NICK_RE = re.compile(r'[\u4e00-\u9fff a-zA-Z0-9]')
# user.id：字母数字和常用符号
_ID_RE = re.compile(r'^[a-zA-Z0-9_\-:.]+$')


def sanitize_nickname(s) -> Optional[str]:
    """校验并规范化昵称。失败返回 None。"""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()[:20]
    if not s:
        return None
    # 拒绝控制字符
    if re.search(r'[\x00-\x1f\x7f]', s):
        return None
    # 至少含 1 个有效字符
    if not _NICK_RE.search(s):
        return None
    return s


def sanitize_user_id(s) -> Optional[str]:
    """校验 user.id。失败返回 None。"""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not (1 <= len(s) <= 64):
        return None
    if not _ID_RE.match(s):
        return None
    return s


def is_safe_url(url: str) -> bool:
    """检查 URL 是否安全（防 SSRF）。"""
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    host = parsed.hostname
    if not host:
        return False

    # 直接是 IP → 检查是否私有
    try:
        ip = ipaddress.ip_address(host)
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved):
            return False
    except ValueError:
        # 是域名 → 允许（实际拉取前可再做 DNS 检查）
        pass

    return True


# MIME 白名单
ALLOWED_MIMES = {
    "image": {"image/png", "image/jpeg", "image/webp"},
    "audio": {"audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg"},
    "video": {"video/mp4", "video/webm"},
    "file": {"application/pdf", "text/plain", "text/markdown"},
}


def check_mime(atype: str, mime: str) -> bool:
    """检查 MIME 是否在白名单内。"""
    allowed = ALLOWED_MIMES.get(atype, set())
    return mime in allowed


def normalize_persona(v) -> str:
    """规范化 persona 字段。"""
    if not v or not isinstance(v, str):
        return "cyrene"
    v = v.strip().lower()
    if v in ("cyrene", "昔涟"):
        return "cyrene"
    if v in ("off", "none", "neutral", "generic", "关闭"):
        return "off"
    return "cyrene"
