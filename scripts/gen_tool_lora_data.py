"""生成 tool LoRA 训练数据（带 <call> 标签）。

用 AMD Radeon Cloud DeepSeek-V4-Flash。
"""
import os, json, time, re, random
from openai import OpenAI

API_KEY = os.environ.get("RADEON_API_KEY", "")
if not API_KEY:
    raise SystemExit("请先 export RADEON_API_KEY=rc-xxx")

client = OpenAI(
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    api_key=API_KEY,
)

SYSTEM = """你正在生成"工具调用"训练数据。格式：

用户：<自然的问题>
助手：<中性语气的回复> <call type="xxx">...</call>

【call 类型】
- <call type="clarify">追问内容</call>      当用户信息不足时
- <call type="suggest">建议内容</call>       当可以主动帮助时
- <call type="action" name="工具名" params='{"key":"val"}'>描述</call>  当需要外部工具时

【可用工具】
- search_weather: {"city": "城市名"}
- search_web: {"query": "搜索词"}
- set_reminder: {"time": "时间", "content": "内容"}
- calculate: {"expression": "数学表达式"}

【核心规则】
1. 语气必须**中性**（"好的"、"我可以帮你"），不要用"呀/呢/♪"等语气词
2. 只有明确需要时才输出 <call>，普通对话不输出
3. <call> 标签在回答末尾
4. 输出严格为：
用户：xxx
助手：xxx <call .../>

不要输出任何解释。"""


# 场景清单
SCENES = {
    "clarify": [
        "用户问某商品价格但没说型号",
        "用户问路线但没说起点终点",
        "用户问症状但没描述具体情况",
        "用户要推荐但没说偏好",
        "用户问时间但没说是哪天",
        "用户问问题但表达很模糊",
        "用户要翻译但没说是哪种语言",
        "用户要代码但没说是哪种语言",
        "用户问天气但没说城市",
        "用户要对比但没说对比哪些",
        "用户问政策但没说地区",
        "用户要摘要但没给原文",
        "用户问价格但没说数量",
        "用户问故障但没说设备型号",
        "用户要方案但没说约束条件",
    ],
    "suggest": [
        "用户说今天很累，可以建议休息",
        "用户说开心，可以建议分享更多",
        "用户说在写作业，可以建议帮忙",
        "用户说要去旅游，可以建议查天气",
        "用户说想学习，可以建议规划",
        "用户说很无聊，可以建议娱乐",
        "用户说要买东西，可以建议比价",
        "用户说生病了，可以建议看医生",
        "用户说赶时间，可以建议快速方案",
        "用户说想做菜，可以建议菜谱",
    ],
    "action": [
        "用户查北京天气",
        "用户查上海天气",
        "用户搜最新新闻",
        "用户搜一首歌的歌词",
        "用户提醒下午3点开会",
        "用户提醒明天交作业",
        "用户算 25 × 37",
        "用户算圆的面积 r=5",
        "用户查深圳天气",
        "用户搜 Python 教程",
    ],
    "negative": [
        "用户普通打招呼",
        "用户说谢谢",
        "用户聊日常",
        "用户表达情绪",
        "用户问简单事实",
        "用户闲聊",
        "用户说再见",
        "用户夸奖助手",
        "用户分享心情",
        "用户问概念",
    ],
}


def call_api(scene, ctype, retries=4):
    user_msg = f"场景：{scene}\n类型：{ctype}\n请生成 5 组对话样本。"
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="DeepSeek-V4-Flash",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=1.0, top_p=0.95, max_tokens=800, timeout=60,
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = 2.0 * (2 ** attempt)
            print(f"[retry {attempt+1}] {str(e)[:80]}, 等 {wait:.0f}s")
            time.sleep(wait)
    return None


def parse_samples(text):
    """从输出里提取 (user, assistant) 对。"""
    if not text:
        return []
    text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    samples = []
    lines = text.split("\n")
    user, assistant = None, None
    for line in lines:
        line = line.strip()
        if line.startswith("用户：") or line.startswith("用户:"):
            user = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        elif line.startswith("助手：") or line.startswith("助手:"):
            assistant = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            if user and assistant:
                if 3 <= len(user) <= 100 and 5 <= len(assistant) <= 300:
                    samples.append({"user": user, "assistant": assistant})
                user, assistant = None, None
    return samples


def main():
    out = []
    t0 = time.time()

    # 遍历三类 call 场景
    for ctype, scenes in SCENES.items():
        for scene in scenes:
            if len([x for x in out if x.get("type") == ctype]) >= (
                150 if ctype == "clarify" else
                100 if ctype == "action" else
                100 if ctype == "suggest" else 150
            ):
                break

            text = call_api(scene, ctype)
            if not text:
                continue
            samples = parse_samples(text)
            for s in samples:
                # 类型校验
                if ctype != "negative" and f'<call type="{ctype}"' not in s["assistant"]:
                    continue
                if ctype == "negative" and "<call" in s["assistant"]:
                    continue

                out.append({
                    "type": ctype,
                    "user": s["user"],
                    "assistant": s["assistant"],
                })
                print(f"[{len(out):>3}] {ctype} | {s['user'][:30]}", flush=True)

                # 每 50 条增量保存
                if len(out) % 50 == 0:
                    with open("/mnt/workspace/data/tool_lora/raw.json", "w",
                              encoding="utf-8") as _f:
                        json.dump(out, _f, ensure_ascii=False, indent=2)
                    print(f"  [save] 增量保存 {len(out)} 条", flush=True)

            time.sleep(0.3)

    print(f"\n[done] {len(out)} 条 | {time.time()-t0:.0f}s")

    # 按类型统计
    from collections import Counter
    cnt = Counter(x["type"] for x in out)
    for k, v in cnt.items():
        print(f"  {k}: {v}")

    # 保存
    with open("/mnt/workspace/data/tool_lora/raw.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[save] /mnt/workspace/data/tool_lora/raw.json")


if __name__ == "__main__":
    main()
