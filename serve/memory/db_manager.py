"""统一数据库管理器：读写都走此入口，避免锁冲突。

- WAL 模式 + busy_timeout
- 每操作开新连接（短事务，安全）
- 线程安全
- 支持读/写/事务
- Web 后台可调用
"""
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, List, Dict, Optional

# ============ DB 路径注册 ============
DBS = {
    "memory":   "data/memory.db",
    "user_kb":  "data/user_kb.db",
    "feedback": "data/feedback.db",
}


def db_path(name: str) -> str:
    if name not in DBS:
        raise ValueError(f"未知数据库: {name}")
    p = DBS[name]
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    return p


def list_dbs() -> List[Dict]:
    out = []
    for name, path in DBS.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        out.append({"name": name, "path": path,
                    "size": size, "exists": exists})
    return out


# ============ 连接 ============
@contextmanager
def connect(name: str, readonly: bool = False):
    """上下文管理：短连接 + WAL + busy_timeout。"""
    path = db_path(name)
    if readonly:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True,
                                timeout=30.0, check_same_thread=False)
    else:
        conn = sqlite3.connect(path, timeout=30.0,
                                check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
    finally:
        conn.close()


# ============ 高层 API ============
def query(name: str, sql: str, params: tuple = ()) -> List[Dict]:
    """查询，返回 dict 列表。"""
    with connect(name, readonly=True) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def query_one(name: str, sql: str, params: tuple = ()) -> Optional[Dict]:
    with connect(name, readonly=True) as conn:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None


def execute(name: str, sql: str, params: tuple = ()) -> int:
    """执行写操作，返回 lastrowid 或 rowcount。"""
    with connect(name, readonly=False) as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid if cur.lastrowid else cur.rowcount


def executemany(name: str, sql: str, seq: List[tuple]) -> int:
    with connect(name, readonly=False) as conn:
        cur = conn.executemany(sql, seq)
        conn.commit()
        return cur.rowcount


@contextmanager
def transaction(name: str):
    """显式事务。"""
    with connect(name, readonly=False) as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


# ============ 表信息 ============
def list_tables(name: str) -> List[Dict]:
    with connect(name, readonly=True) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "ORDER BY name"
        ).fetchall()
        out = []
        for r in rows:
            tname = r["name"]
            try:
                cnt = conn.execute(
                    f"SELECT count(*) FROM `{tname}`").fetchone()[0]
                cols = [c["name"] for c in conn.execute(
                    f"PRAGMA table_info(`{tname}`)").fetchall()]
                out.append({"name": tname, "count": cnt, "columns": cols})
            except Exception as e:
                out.append({"name": tname, "count": -1,
                            "error": str(e)})
        return out


def table_columns(name: str, table: str) -> List[str]:
    with connect(name, readonly=True) as conn:
        rows = conn.execute(f"PRAGMA table_info(`{table}`)").fetchall()
        return [r["name"] for r in rows]


def fetch_rows(name: str, table: str, offset: int = 0, limit: int = 50,
               order_by: Optional[str] = None, desc: bool = True,
               where: Optional[str] = None,
               where_params: tuple = ()) -> Dict:
    """分页查询，带白名单校验。"""
    valid = [t["name"] for t in list_tables(name)]
    if table not in valid:
        raise ValueError(f"表不存在: {table}")

    order_clause = ""
    if order_by:
        cols = table_columns(name, table)
        if order_by in cols:
            order_clause = f" ORDER BY `{order_by}` {'DESC' if desc else 'ASC'}"

    where_clause = ""
    if where:
        # where 由调用方保证安全（只用列名 + 占位符）
        where_clause = f" WHERE {where}"

    with connect(name, readonly=True) as conn:
        total = conn.execute(
            f"SELECT count(*) FROM `{table}`{where_clause}",
            where_params
        ).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM `{table}`{where_clause}{order_clause} "
            f"LIMIT ? OFFSET ?",
            where_params + (limit, offset)
        ).fetchall()
        return {
            "db": name, "table": table, "total": total,
            "offset": offset, "limit": limit,
            "items": [dict(r) for r in rows],
        }


# ============ 变更通知（实时刷新） ============
class ChangeNotifier:
    """简单版本号机制：每次写操作 +1，前端轮询版本号决定是否刷新。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._versions = {name: 0 for name in DBS}

    def bump(self, name: str):
        with self._lock:
            self._versions[name] = self._versions.get(name, 0) + 1

    def get(self, name: str = None) -> Dict:
        with self._lock:
            if name:
                return {"name": name,
                        "version": self._versions.get(name, 0)}
            return dict(self._versions)


_notifier = ChangeNotifier()


def notify(name: str):
    _notifier.bump(name)


def get_versions() -> Dict:
    return _notifier.get()
