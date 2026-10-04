"""只生成 negative 类型数据（普通对话，不触发 call）。"""
import os, json, time, re
from openai import OpenAI

client = OpenAI(
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    api_key=os.environ["RADEON_API_KEY"],
)

SYSTEM = """你正在生成"普通对话"训练数据。

格式严格为：
用户：xxx
助手：xxx

【核心规则】
1. 助手回复必须**中性、简洁**（"好的"、"我明白了"、"这样啊"）
2. **绝对不能**出现 <call> 标签
3. **不能**追问、不能建议、不能调用工具
4. 回复长度 5-30 字
5. 输出严格为：
用户：xxx
助手：xxx
不要任何解释。"""

SCENES = [
    "用户打招呼", "用户说谢谢", "用户说再见", "用户说好的",
    "用户说嗯", "用户说收到", "用户说知道了", "用户说好的谢谢",
    "用户分享心情好", "用户分享心情不好", "用户说今天不错",
    "用户说天气好", "用户说吃过了", "用户说在忙", "用户说等一下",
    "用户说继续", "用户说然后呢", "用户说不错", "用户说有意思",
    "用户说真的吗", "用户说太好了", "用户说可惜", "用户说算了",
    "用户说没事", "用户说没关系", "用户说辛苦了", "用户说晚安",
    "用户说早安", "用户问几点了", "用户说在干嘛",
]

def call_api(scene):
    try:
        resp = client.chat.completions.create(
            model="DeepSeek-V4-Flash",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"场景：{scene}\n请生成 5 组对话。"},
            ],
            temperature=1.0, top_p=0.95, max_tokens=500, timeout=60,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[err] {str(e)[:80]}", flush=True)
        return None

def parse(text):
    if not text:
        return []
    text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    samples = []
    user, assistant = None, None
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("用户："):
            user = line.split("：", 1)[-1].strip()
        elif line.startswith("助手："):
            assistant = line.split("：", 1)[-1].strip()
            if user and assistant:
                samples.append({"user": user, "assistant": assistant})
            user, assistant = None, None
    return samples

def main():
    p = "/mnt/workspace/data/tool_lora/raw.json"
    with open(p) as f:
        existing = json.load(f)

    out = [x for x in existing if x.get("type") == "negative"]
    print(f"[start] 现有 negative: {len(out)} 条", flush=True)

    seen = set(x["user"] for x in out)
    t0 = time.time()

    for scene in SCENES:
        if len(out) >= 150:
            break
        text = call_api(scene)
        if not text:
            time.sleep(1)
            continue
        for s in parse(text):
            if "<call" in s["assistant"]:
                continue
            if not (2 <= len(s["user"]) <= 30
                    and 2 <= len(s["assistant"]) <= 40):
                continue
            if s["user"] in seen:
                continue
            seen.add(s["user"])
            out.append({"type": "negative", **s})
            print(f"[{len(out):>3}] {s['user'][:30]}", flush=True)
            if len(out) % 30 == 0:
                merged = [x for x in existing if x.get("type") != "negative"]
                merged += out
                with open(p, "w") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=2)
        time.sleep(0.3)

    merged = [x for x in existing if x.get("type") != "negative"]
    merged += out
    with open(p, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n[done] negative={len(out)} 条 | {time.time()-t0:.0f}s")
    from collections import Counter
    print("=== 总分布 ===")
    for k, v in Counter(x["type"] for x in merged).items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
