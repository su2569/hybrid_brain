"""Web 管理后台认证：首次引导设置密码 + 加盐哈希。

存储：data/admin.json
密码：pbkdf2_sha256, 200k iterations
Token：hmac 派生自密码 hash
"""
import os
import json
import time
import hmac
import base64
import hashlib
import secrets
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Response, HTTPException, Cookie, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

router = APIRouter()

# ========== 配置 ==========
ADMIN_FILE = Path("data/admin.json")
TEMPLATES = Path(__file__).parent / "templates"
TOKEN_MAX_AGE = 86400 * 7     # 7 天
PBKDF2_ITER = 200_000

# ========== 限速 ==========
_LOGIN_ATTEMPTS = {}   # ip -> [timestamps]
RATE_WINDOW = 60       # 秒
RATE_MAX = 5           # 窗口内最多尝试次数


def _rate_check(ip: str) -> bool:
    now = time.time()
    hist = _LOGIN_ATTEMPTS.get(ip, [])
    hist = [t for t in hist if now - t < RATE_WINDOW]
    _LOGIN_ATTEMPTS[ip] = hist
    return len(hist) < RATE_MAX


def _rate_record(ip: str):
    _LOGIN_ATTEMPTS.setdefault(ip, []).append(time.time())


# ========== 密码哈希 ==========
def _hash_password(password: str, salt: bytes = None) -> dict:
    """返回 {salt, hash}。"""
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITER)
    return {
        "salt": salt.hex(),
        "hash": dk.hex(),
    }


def _verify_password(password: str, record: dict) -> bool:
    salt = bytes.fromhex(record.get("salt", ""))
    expect_hash = record.get("hash", "")
    iters = int(record.get("iterations", PBKDF2_ITER))
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iters)
    return hmac.compare_digest(dk.hex(), expect_hash)


def _load_admin() -> Optional[dict]:
    if not ADMIN_FILE.exists():
        return None
    try:
        return json.loads(ADMIN_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[admin-load-error] {e}", flush=True)
        return None


def _save_admin(record: dict):
    ADMIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ADMIN_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(ADMIN_FILE)
    try:
        os.chmod(ADMIN_FILE, 0o600)
    except Exception:
        pass


def _is_initialized() -> bool:
    return _load_admin() is not None


# ========== Token ==========
def _secret_key() -> bytes:
    """从 admin.json 派生签名密钥。"""
    rec = _load_admin()
    if not rec:
        return b"__uninit__"
    return hashlib.sha256(
        (rec["salt"] + rec["hash"]).encode()).digest()


def _make_token() -> str:
    ts = str(int(time.time()))
    sig = hmac.new(_secret_key(), ts.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{ts}.{sig}"


def _verify_token(token: Optional[str]) -> bool:
    if not token or "." not in token:
        return False
    try:
        ts, sig = token.split(".", 1)
        if int(time.time()) - int(ts) > TOKEN_MAX_AGE:
            return False
        expect = hmac.new(_secret_key(), ts.encode(),
                          hashlib.sha256).hexdigest()[:16]
        return hmac.compare_digest(sig, expect)
    except Exception:
        return False


def require_auth(admin_token: str = Cookie(None)):
    if not _verify_token(admin_token):
        raise HTTPException(401, "未登录")


# ========== Setup Token（首次初始化用） ==========
_SETUP_TOKEN = None


def get_setup_token() -> str:
    """获取（或生成）setup token。仅在未初始化时有效。"""
    global _SETUP_TOKEN
    if _is_initialized():
        return ""
    if _SETUP_TOKEN is None:
        # 环境变量优先
        env = os.environ.get("HB_SETUP_TOKEN", "").strip()
        _SETUP_TOKEN = env if env else secrets.token_urlsafe(16)
        print("═" * 60, flush=True)
        print(f"[admin] 首次启动，尚未设置密码", flush=True)
        print(f"[admin] 请访问: http://<host>:8000/admin/setup", flush=True)
        print(f"[admin] SETUP TOKEN: {_SETUP_TOKEN}", flush=True)
        print("═" * 60, flush=True)
    return _SETUP_TOKEN


def _verify_setup_token(token: str) -> bool:
    if _is_initialized():
        return False
    return bool(_SETUP_TOKEN and hmac.compare_digest(token, _SETUP_TOKEN))


# ========== 路由 ==========
@router.get("/", response_class=HTMLResponse)
def admin_root(admin_token: str = Cookie(None)):
    if not _is_initialized():
        return RedirectResponse("/admin/setup")
    if not _verify_token(admin_token):
        return RedirectResponse("/admin/login")
    return RedirectResponse("/admin/dashboard")


@router.get("/setup", response_class=HTMLResponse)
def setup_page():
    if _is_initialized():
        return RedirectResponse("/admin/login")
    html = (TEMPLATES / "setup.html").read_text(encoding="utf-8")
    return html


@router.post("/setup")
async def setup_submit(
    request: Request,
    setup_token: str = Form(...),
    password: str = Form(...),
    password2: str = Form(...),
):
    """首次设置密码。"""
    if _is_initialized():
        raise HTTPException(400, "已初始化，无法重复设置")
    if not _verify_setup_token(setup_token):
        raise HTTPException(401, "SETUP TOKEN 错误")
    if len(password) < 8:
        raise HTTPException(400, "密码至少 8 位")
    if password != password2:
        raise HTTPException(400, "两次密码不一致")

    rec = _hash_password(password)
    rec.update({
        "algo": "pbkdf2_sha256",
        "iterations": PBKDF2_ITER,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    })
    _save_admin(rec)

    # 签发 token
    token = _make_token()
    resp = JSONResponse({"ok": True})
    resp.set_cookie("admin_token", token, httponly=True,
                    max_age=TOKEN_MAX_AGE, samesite="lax")
    print(f"[admin] 密码已设置", flush=True)
    return resp


@router.get("/login", response_class=HTMLResponse)
def login_page():
    if not _is_initialized():
        return RedirectResponse("/admin/setup")
    return (TEMPLATES / "login.html").read_text(encoding="utf-8")


@router.post("/login")
def login(request: Request, response: Response, password: str = Form(...)):
    if not _is_initialized():
        raise HTTPException(400, "尚未初始化")

    ip = request.client.host if request.client else "?"
    if not _rate_check(ip):
        raise HTTPException(429, "尝试过于频繁，请稍后再试")

    rec = _load_admin()
    if not _verify_password(password, rec):
        _rate_record(ip)
        raise HTTPException(401, "密码错误")

    token = _make_token()
    resp = JSONResponse({"ok": True})
    resp.set_cookie("admin_token", token, httponly=True,
                    max_age=TOKEN_MAX_AGE, samesite="lax")
    return resp


@router.post("/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("admin_token")
    return resp


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(admin_token: str = Cookie(None)):
    if not _is_initialized():
        return RedirectResponse("/admin/setup")
    if not _verify_token(admin_token):
        return RedirectResponse("/admin/login")
    return (TEMPLATES / "admin.html").read_text(encoding="utf-8")


@router.get("/profile", response_class=HTMLResponse)
def profile_page(admin_token: str = Cookie(None)):
    if not _verify_token(admin_token):
        return RedirectResponse("/admin/login")
    return (TEMPLATES / "profile.html").read_text(encoding="utf-8")


@router.post("/change_password")
def change_password(
    old_password: str = Form(...),
    new_password: str = Form(...),
    new_password2: str = Form(...),
    admin_token: str = Cookie(None),
):
    """登录后修改密码。"""
    if not _verify_token(admin_token):
        raise HTTPException(401, "未登录")
    rec = _load_admin()
    if not _verify_password(old_password, rec):
        raise HTTPException(401, "旧密码错误")
    if len(new_password) < 8:
        raise HTTPException(400, "新密码至少 8 位")
    if new_password != new_password2:
        raise HTTPException(400, "两次密码不一致")

    new_rec = _hash_password(new_password)
    new_rec.update({
        "algo": rec["algo"],
        "iterations": rec["iterations"],
        "created_at": rec.get("created_at", int(time.time())),
        "updated_at": int(time.time()),
    })
    _save_admin(new_rec)

    # 换新 token（旧 token 失效）
    token = _make_token()
    resp = JSONResponse({"ok": True})
    resp.set_cookie("admin_token", token, httponly=True,
                    max_age=TOKEN_MAX_AGE, samesite="lax")
    return resp
