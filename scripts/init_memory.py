"""初始化 L1 会话记忆 SQLite。"""
import os, sqlite3

DB = "data/memory.db"
os.makedirs(os.path.dirname(DB), exist_ok=True)

conn = sqlite3.connect(DB)
conn.executescript("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id, ts);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.6,
    created_at INTEGER NOT NULL,
    source TEXT DEFAULT 'rule'
);
CREATE INDEX IF NOT EXISTS idx_user_facts ON facts(user_id, created_at DESC);
""")
conn.commit()
conn.close()
print(f"[ok] {DB} 初始化完成")
