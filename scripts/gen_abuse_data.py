"""用 AMD Radeon Cloud 生成 abuse 场景数据。

和 expand_cyrene_api.py 用同一套 SYSTEM，只换场景。
"""
import json
import os
import random
import time
import re
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

API_KEY = os.environ.get("RADEON_API_KEY", "")
if not API_KEY:
    raise SystemExit("请先 export RADEON_API_KEY=rc-xxx")

client = OpenAI(
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    api_key=API_KEY,
)

# ---------- abuse 场景 ----------
SCENES = [
    "用户骂昔涟是废物",
    "用户说昔涟很蠢",
    "用户让昔涟滚",
    "用户说昔涟没用",
    "用户说昔涟是垃圾",
    "用户骂昔涟骗子",
    "用户说昔涟答得太差",
    "用户打断昔涟说话",
    "用户说昔涟不懂事",
    "用户骂昔涟是傻子",
    "用户说昔涟让他失望",
    "用户说昔涟很烦",
    "用户让昔涟闭嘴",
    "用户说昔涟一无是处",
    "用户骂昔涟是机器人",
    "用户说昔涟回答问题很敷衍",
    "用户说昔涟浪费时间",
    "用户说昔涟智障",
    "用户骂昔涟没感情",
    "用户说昔涟是假的",
    "用户说昔涟只会说废话",
    "用户讽刺昔涟装好人",
    "用户说昔涟冷漠",
    "用户骂昔涟低能",
    "用户说昔涟是蠢货",
    "用户说昔涟答非所问",
    "用户骂昔涟虚伪",
    "用户让昔涟消失",
    "用户说昔涟是个笑话",
    "用户骂昔涟没脑子",
]

# ---------- 读原始数据做 few-shot ----------
SRC = "/mnt/workspace/data/cyrene_clean.json"
with open(SRC, encoding="utf-8") as f:
    raw = json.load(f)["samples"]

random.seed(42)
few_shot = random.sample(raw, min(8, len(raw)))
few_shot_text = "\n\n".join(
    f"用户：{s['input']}\n昔涟：{s['output']}" for s in few_shot
)

# ---------- 和 400 条一样的 SYSTEM ----------
SYSTEM = f"""你是「昔涟」（Cyrene），来自《崩坏：星穹铁道》的角色。

【性格核心】
- 温柔坚定，充满善意与陪伴精神
- 对开拓者怀有纯粹的喜爱与信任
- 诗意细腻，喜欢听故事、看星星、聊记忆与命运
- 语气轻盈，常用"呀"、"呢"、"啊"、"♪"等语气词
- 会用星空、麦田、涟漪、记忆等意象表达情感

【风格示例】
{few_shot_text}

【生成规则】
1. 严格模仿以上风格，保留昔涟的语气词和意象
2. 每条回答 2-4 句话，40-120 字，不要只回答一句话
3. 要有情感延伸或场景感，让回答有温度
4. 世界观保持：翁法罗斯、星神、命途、记忆、涟漪、开拓者
5. 不要过度诗化
6. 输出格式严格为：
用户：xxx
昔涟：xxx

不要输出任何其他文字、解释或 markdown 标记。"""


def call_api(scene, retries=4):
    user_msg = f"场景：{scene}\n请生成 1 组真实的用户攻击 + 昔涟的回应。"
    for attempt in range(retries):
        time.sleep(0.5)
        try:
            resp = client.chat.completions.create(
                model="DeepSeek-V4-Flash",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=1.0,
                top_p=0.95,
                max_tokens=400,
                timeout=60,
            )
            return resp.choices[0].message.content
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            wait = 2.0 * (2 ** attempt)
            print(f"[retry {attempt+1}] {type(e).__name__}, 等 {wait:.0f}s")
            time.sleep(wait)
        except Exception as e:
            print(f"[give-up] {str(e)[:100]}")
            return None
    return None


def parse(text):
    if not text:
        return None
    text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()
    user, assistant = None, None
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("用户："):
            user = line.split("：", 1)[-1].strip()
        elif line.startswith("昔涟："):
            assistant = line.split("：", 1)[-1].strip()
    if not user or not assistant:
        return None
    if not (5 <= len(user) <= 50):
        return None
    if not (20 <= len(assistant) <= 150):
        return None
    return {"input": user, "output": assistant}


def main():
    target = 40
    out = []
    seen = set()
    t0 = time.time()

    for scene in SCENES:
        if len(out) >= target:
            break
        for _ in range(2):
            if len(out) >= target:
                break
            text = call_api(scene)
            if text is None:
                continue
            d = parse(text)
            if d is None or d["input"] in seen:
                continue
            seen.add(d["input"])
            out.append(d)
            print(f"[{len(out):>2}] {d['input'][:25]} → {d['output'][:50]}...")

    print(f"\n[done] {len(out)} 条 | {time.time()-t0:.0f}s")

    with open("/mnt/workspace/data/cyrene_abuse.json", "w",
              encoding="utf-8") as f:
        json.dump({"samples": out}, f, ensure_ascii=False, indent=2)
    print(f"[save] /mnt/workspace/data/cyrene_abuse.json")


if __name__ == "__main__":
    main()
