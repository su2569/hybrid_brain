"""L1 会话记忆测试。"""
import requests, json

BASE = "http://localhost:8000/chat"
SID = "test-session-001"
UID = "test-user-001"


def chat(q, session_id=SID, user_id=UID):
    r = requests.post(BASE, json={
        "query": q,
        "session_id": session_id,
        "user": {"id": user_id},
    })
    d = r.json()
    print(f"\n>>> {q}")
    print(f"    {d['answer'][:120]}")
    return d


print("=" * 60)
print("【1】告诉昔涟事实")
chat("我喜欢喝美式，不加糖")
chat("我住在杭州")
chat("我叫小明")

print("\n" + "=" * 60)
print("【2】追问事实（不同 session）")
chat("我喜欢喝什么咖啡", session_id="test-session-002")

print("\n" + "=" * 60)
print("【3】同 session 上下文连续")
chat("我刚才说了什么", session_id="test-session-003")
chat("那你还记得吗", session_id="test-session-003")

print("\n" + "=" * 60)
print("【4】另一个用户（不应看到 test-user-001 的事实）")
chat("我喜欢喝什么咖啡", session_id="test-session-004", user_id="other-user")
