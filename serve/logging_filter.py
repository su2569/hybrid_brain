"""抑制高频轮询的 access log。"""
import logging


class AccessLogFilter(logging.Filter):
    """过滤 /admin/api/db/fingerprint 等高频轮询。"""

    SILENT_PATHS = (
        "/admin/api/db/fingerprint",
        "/health",
        "/favicon.ico",
    )

    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        for p in self.SILENT_PATHS:
            if p in msg:
                return False
        return True


def install():
    """把 filter 挂到 uvicorn access logger。"""
    lg = logging.getLogger("uvicorn.access")
    for h in lg.handlers:
        h.addFilter(AccessLogFilter())
    print("[ok] access log filter installed", flush=True)
