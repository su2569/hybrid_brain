"""生成情感支持类 chat 数据。"""
import os, json, time
from openai import OpenAI

key = os.environ.get("RADEON_API_KEY", "").strip()
print(f"[key] 长度 {len(key)}, 前缀 {key[:8]}...", flush=True)

client = OpenAI(
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    api_key=key,
)

SCENES = [
    "我有点难过", "我很累", "有点烦", "不开心", "压力好大",
    "想哭", "心里堵得慌", "陪我聊聊", "想找人说说话",
    "在吗", "忙吗", "今天糟透了", "坚持不下去", "好焦虑",
    "睡不着", "累到不想说话", "情绪低落", "想放弃",
    "最近很迷茫", "感觉很孤独",
]

SYSTEM = """生成 10 条真实用户输入，场景是情感倾诉/闲聊。
每条 5-25 字，口语化，像真人打字。
不要提问，不要知识型问题，纯情感表达。
每行 1 条，不要编号，不要引号。"""

all_items = []
for i, scene in enumerate(SCENES, 1):
    print(f"[{i}/20] {scene}", flush=True)
    try:
        resp = client.chat.completions.create(
            model="DeepSeek-V4-Flash",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"场景：{scene}"},
            ],
            temperature=1.1, top_p=0.95, max_tokens=500,
            timeout=30,
        )
        text = resp.choices[0].message.content
        n = 0
        for line in text.strip().split("\n"):
            line = line.strip().lstrip("0123456789.、- ")
            if 3 <= len(line) <= 30:
                all_items.append({"text": line, "intent": "chat"})
                n += 1
        print(f"  → {n} 条", flush=True)
    except Exception as e:
        print(f"  [fail] {str(e)[:100]}", flush=True)
    time.sleep(0.3)

seen = set()
uniq = []
for x in all_items:
    if x["text"] not in seen:
        seen.add(x["text"])
        uniq.append(x)

print(f"\n[total] {len(uniq)} 条", flush=True)

with open("/mnt/workspace/data/chat_extra.json", "w", encoding="utf-8") as f:
    json.dump({"samples": uniq}, f, ensure_ascii=False, indent=2)
print("[save] /mnt/workspace/data/chat_extra.json", flush=True)
