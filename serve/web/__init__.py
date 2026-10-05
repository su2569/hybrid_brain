"""Web 管理后台路由注册。"""
from fastapi import APIRouter
from .auth import router as auth_router
from .api import kb, db, memory, graph, users, feedback

admin_router = APIRouter(prefix="/admin", tags=["admin"])
admin_router.include_router(auth_router)
admin_router.include_router(kb.router, prefix="/api/kb")
admin_router.include_router(db.router, prefix="/api/db")
admin_router.include_router(memory.router, prefix="/api/memory")
admin_router.include_router(graph.router, prefix="/api/graph")
admin_router.include_router(users.router, prefix="/api/users")
admin_router.include_router(feedback.router, prefix="/api/feedback")


# ============================================================
# 启动时触发首次初始化检查（打印 SETUP TOKEN）
# ============================================================
from .auth import get_setup_token as _trigger_setup_check
_trigger_setup_check()
