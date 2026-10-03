"""生成「指令跟随」型昔涟数据，用于增强 LoRA 对 prompt 的服从性。"""
import os, json, time, random
from openai import OpenAI

client = OpenAI(
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    api_key=os.environ["RADEON_API_KEY"],
)

# 5 种约束 × 6 个 query = 30 组
CONSTRAINTS = [
    ("用不超过 20 字回答。", "简洁"),
    ("回答必须以问号结尾。", "问号"),
    ("直接回答，不写诗，不堆意象。", "直白"),
    ("禁止猜测，禁止假设。", "禁猜"),
    ("资料里没答案时直接说'不知道'。", "承认"),
]

QUERIES = [
    "我一般喝什么咖啡",
    "我喜欢什么颜色",
    "我住在哪里",
    "你觉得我性格怎样",
    "我今天心情不好",
    "你是谁",
]

# 生成 prompt
GEN_SYSTEM = """你正在生成"昔涟"角色的训练数据。昔涟性格温柔诗意。

我会给你一个 system prompt 和一个 user 输入，请生成 1 条 assistant 回答。

【关键要求】
1. 严格遵守 system prompt 的约束
2. 如果 system 说"简洁"，就真的简短
3. 如果 system 说"必须有问号"，回答必须带问号
4. 保持昔涟人格（温柔/亲近），但服从指令

【输出格式】
只输出 assistant 回答原文，不要任何解释。"""

items = []
for constraint, tag in CONSTRAINTS:
    for q in QUERIES:
        user_msg = f"system prompt：你是昔涟。{constraint}\nuser：{q}"
        try:
            resp = client.chat.completions.create(
                model="DeepSeek-V4-Flash",
                messages=[
                    {"role": "system", "content": GEN_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=1.0, top_p=0.95, max_tokens=200, timeout=30,
            )
            ans = resp.choices[0].message.content.strip()
            # 清理
            ans = ans.strip('"').strip("「」").strip()
            if 5 <= len(ans) <= 100:
                items.append({
                    "system": f"你是昔涟。{constraint}",
                    "user": q,
                    "assistant": ans,
                    "tag": tag,
                })
                print(f"[{len(items):>2}] {tag} | {q} → {ans[:50]}")
        except Exception as e:
            print(f"[fail] {str(e)[:80]}")
        time.sleep(0.3)

# 去重
seen = set()
uniq = []
for x in items:
    key = (x["user"], x["tag"])
    if key not in seen:
        seen.add(key)
        uniq.append(x)

print(f"\n[total] {len(uniq)} 条")

with open("/mnt/workspace/data/cyrene_constrained.json", "w",
          encoding="utf-8") as f:
    json.dump({"samples": uniq}, f, ensure_ascii=False, indent=2)
print(f"[save] cyrene_constrained.json")
