"""SQLite 只读连接工具（避免写锁冲突）。"""
import os
import sqlite3


def read_conn(path: str, timeout: float = 30.0) -> sqlite3.Connection:
    """只读打开 + 长超时（避免与写入进程冲突）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # 用 URI 模式打开（mode=ro 保证只读，不升级锁）
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout,
                            check_same_thread=False)
    # WAL 下读不阻塞写
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    conn.row_factory = sqlite3.Row
    return conn
