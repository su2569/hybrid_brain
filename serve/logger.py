"""HybridBrain 日志：终端彩色 + 文件纯文本双输出。

规则：
- 终端（stdout）：若 isatty() → ANSI 彩色，否则纯文本
- 文件（logs/hybridbrain.log）：永远纯文本，可 grep
- 环境变量：
    NO_COLOR=1     禁用终端颜色
    FORCE_COLOR=1  强制终端颜色（即使重定向到管道）
    HB_LOG_FILE    日志文件路径（默认 logs/hybridbrain.log）
    HB_LOG_LEVEL   文件日志级别（默认 INFO）
"""
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Any


# ==================== 终端颜色 ====================
def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


_TTY_COLOR = _supports_color()


def _c(code: str) -> str:
    return code if _TTY_COLOR else ""


class C:
    RESET = _c("\033[0m")
    BOLD = _c("\033[1m")
    DIM = _c("\033[2m")
    RED = _c("\033[31m")
    GREEN = _c("\033[32m")
    YELLOW = _c("\033[33m")
    BLUE = _c("\033[34m")
    MAGENTA = _c("\033[35m")
    CYAN = _c("\033[36m")
    WHITE = _c("\033[37m")
    GRAY = _c("\033[90m")
    BR = _c("\033[91m")
    BG = _c("\033[92m")
    BY = _c("\033[93m")
    BB = _c("\033[94m")
    BM = _c("\033[95m")
    BC = _c("\033[96m")


_LEVEL_COLOR = {
    "DEBUG": C.GRAY,
    "INFO":  C.BB,
    "OK":    C.BG,
    "WARN":  C.BY,
    "ERROR": C.BR,
    "FATAL": C.BR + C.BOLD,
}

_TAG_COLOR = {
    "ROUTE":    C.BY,
    "CHAT":     C.BM,
    "RAG":      C.BC,
    "TOOL":     C.BB,
    "PERSONAL": C.BM,
    "SAFETY":   C.BR,
    "REFUSE":   C.BR,
    "ABUSE":    C.BR,
    "MEM":      C.BG,
    "L1":       C.BG,
    "L2":       C.BG,
    "USER_KB":  C.BG,
    "GEN":      C.CYAN,
    "FINALIZE": C.CYAN,
    "API":      C.WHITE,
    "HBP":      C.WHITE,
    "OAI":      C.WHITE,
    "LOAD":     C.GRAY,
    "_":        C.WHITE,
}


# ==================== 文件 logger ====================
_FILE_LOGGER = None


class _CtxFilter(logging.Filter):
    """把 tag 塞进 record 供 formatter 用。"""
    def filter(self, record):
        if not hasattr(record, "tag"):
            record.tag = "-"
        return True


def _init_file_logger():
    global _FILE_LOGGER
    if _FILE_LOGGER is not None:
        return _FILE_LOGGER

    path = os.environ.get("HB_LOG_FILE", "logs/hybridbrain.log")
    level = os.environ.get("HB_LOG_LEVEL", "INFO").upper()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    lg = logging.getLogger("hybridbrain")
    lg.setLevel(getattr(logging, level, logging.INFO))
    lg.propagate = False

    # 避免重复挂 handler
    if not lg.handlers:
        # 轮转：单文件 20MB，保留 5 个
        fh = RotatingFileHandler(
            path, maxBytes=20 * 1024 * 1024,
            backupCount=5, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)-5s] [%(tag)-8s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        fh.addFilter(_CtxFilter())
        lg.addHandler(fh)

    _FILE_LOGGER = lg
    return lg


# ==================== 输出 ====================
def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(level: str, tag: str, msg: Any):
    level = level.upper()
    tag = tag.upper()
    txt = str(msg)

    # ---------- 1. 终端 ----------
    lc = _LEVEL_COLOR.get(level, "")
    tc = _TAG_COLOR.get(tag, C.WHITE)
    line = (
        f"{C.DIM}[{_ts()}]{C.RESET} "
        f"{lc}[{level:5s}]{C.RESET} "
        f"{tc}[{tag:8s}]{C.RESET} "
        f"{txt}"
    )
    try:
        print(line, flush=True)
    except Exception:
        pass

    # ---------- 2. 文件 ----------
    try:
        lg = _init_file_logger()
        lvl = getattr(logging, level if level != "OK" else "INFO",
                      logging.INFO)
        lg.log(lvl, txt, extra={"tag": tag})
    except Exception as e:
        # 文件失败不影响终端
        if os.environ.get("HB_LOG_DEBUG"):
            print(f"[logger-file-error] {e}", file=sys.stderr)


# ==================== 公共接口 ====================
def debug(tag: str, msg: Any): _log("DEBUG", tag, msg)
def info(tag: str, msg: Any):  _log("INFO",  tag, msg)
def ok(tag: str, msg: Any):    _log("OK",    tag, msg)
def warn(tag: str, msg: Any):  _log("WARN",  tag, msg)
def error(tag: str, msg: Any): _log("ERROR", tag, msg)
def fatal(tag: str, msg: Any): _log("FATAL", tag, msg)


# ==================== 语义化快捷 ====================
def route(msg):     info("ROUTE", msg)
def chat(msg):      info("CHAT", msg)
def rag(msg):       info("RAG", msg)
def tool(msg):      info("TOOL", msg)
def personal(msg):  info("PERSONAL", msg)
def safety(msg):    warn("SAFETY", msg)
def refuse(msg):    warn("REFUSE", msg)
def abuse(msg):     warn("ABUSE", msg)
def mem(msg):       ok("MEM", msg)
def l1(msg):        ok("L1", msg)
def l2(msg):        ok("L2", msg)
def gen(msg):       info("GEN", msg)
def finalize(msg):  info("FINALIZE", msg)
def api(msg):       info("API", msg)
def load(msg):      info("LOAD", msg)


# ==================== 启动时打一行 banner ====================
def banner():
    """服务启动时打一行，标记新会话。"""
    info("LOAD", f"logger 就绪 tty_color={_TTY_COLOR} "
                 f"file={os.environ.get('HB_LOG_FILE', 'logs/hybridbrain.log')}")
