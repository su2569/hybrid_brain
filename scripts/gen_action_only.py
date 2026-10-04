"""只生成 action 类型数据（工具调用）。

目标 100+ 条，追加到现有 raw.json。
"""
import os, json, time, re
from openai import OpenAI

API_KEY = os.environ.get("RADEON_API_KEY", "")
if not API_KEY:
    raise SystemExit("export RADEON_API_KEY=rc-xxx")

client = OpenAI(
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    api_key=API_KEY,
)

SYSTEM = """你正在生成"工具调用"训练数据。

格式严格为：
用户：xxx
助手：好的，我帮你处理。<call type="action" name="工具名" params='{"key":"val"}'>简短描述</call>

【可用工具】
- search_weather: {"city": "城市名"}
- search_web: {"query": "搜索词"}
- set_reminder: {"time": "时间", "content": "内容"}
- calculate: {"expression": "数学表达式"}
- currency_convert: {"amount": "金额", "from": "币种", "to": "币种"}

【核心规则】
1. 语气**中性**（"好的"、"我帮你"），不要用"呀/呢/♪"
2. 必须在回答末尾输出 <call> 标签
3. params 必须是合法 JSON
4. 输出严格为：
用户：xxx
助手：xxx<call .../>

不要任何解释、不要 markdown。"""

# 50 个 action 场景（覆盖多种工具）
SCENES = [
    # 天气
    ("用户查北京天气", "search_weather"),
    ("用户查上海天气", "search_weather"),
    ("用户查深圳天气", "search_weather"),
    ("用户查杭州天气", "search_weather"),
    ("用户查广州天气", "search_weather"),
    ("用户查成都天气", "search_weather"),
    ("用户查明天天气", "search_weather"),
    ("用户查周末天气", "search_weather"),
    # 搜索
    ("用户搜最新新闻", "search_web"),
    ("用户搜一首歌的歌词", "search_web"),
    ("用户搜 Python 教程", "search_web"),
    ("用户搜菜谱", "search_web"),
    ("用户搜电影推荐", "search_web"),
    ("用户搜附近餐厅", "search_web"),
    ("用户搜今天的汇率", "search_web"),
    ("用户搜驾考流程", "search_web"),
    ("用户搜演唱会门票", "search_web"),
    ("用户搜考研资料", "search_web"),
    # 提醒
    ("用户提醒下午3点开会", "set_reminder"),
    ("用户提醒明天交作业", "set_reminder"),
    ("用户提醒周日给妈妈打电话", "set_reminder"),
    ("用户提醒明天下午2点取快递", "set_reminder"),
    ("用户提醒一小时后喝水", "set_reminder"),
    ("用户提醒我月底交房租", "set_reminder"),
    ("用户提醒晚上8点吃药", "set_reminder"),
    ("用户提醒周五看电影", "set_reminder"),
    # 计算
    ("用户算 25 × 37", "calculate"),
    ("用户算 128 + 256", "calculate"),
    ("用户算 999 × 999", "calculate"),
    ("用户算 3 的 10 次方", "calculate"),
    ("用户算 2 的 20 次方", "calculate"),
    ("用户算 456 ÷ 12", "calculate"),
    ("用户算 88 × 99", "calculate"),
    ("用户算 17 的平方", "calculate"),
    # 汇率
    ("用户算 100 美元等于多少人民币", "currency_convert"),
    ("用户算 500 人民币等于多少美元", "currency_convert"),
    ("用户算 1000 日元等于多少人民币", "currency_convert"),
    ("用户算 200 欧元等于多少美元", "currency_convert"),
    ("用户算 5000 港币等于多少人民币", "currency_convert"),
    # 更多
    ("用户查南京天气", "search_weather"),
    ("用户查西安天气", "search_weather"),
    ("用户搜英文单词发音", "search_web"),
    ("用户搜股票行情", "search_web"),
    ("用户提醒明早7点起床", "set_reminder"),
    ("用户算 1234 × 5678", "calculate"),
    ("用户算 100 英镑等于多少人民币", "currency_convert"),
    ("用户搜附近的加油站", "search_web"),
    ("用户查武汉天气", "search_weather"),
    ("用户算 15% 的 200", "calculate"),
]


def call_api(scene, tool):
    user_msg = f"场景：{scene}\n工具：{tool}\n请生成 4 组对话样本。"
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
        print(f"[err] {str(e)[:80]}", flush=True)
        return None


def parse(text):
    """提取 (user, assistant) 对。"""
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
    # 加载现有数据
    existing = []
    p = "/mnt/workspace/data/tool_lora/raw.json"
    if os.path.exists(p):
        with open(p) as f:
            existing = json.load(f)
        print(f"[load] 现有 {len(existing)} 条", flush=True)

    out = [x for x in existing if x.get("type") == "action"]
    print(f"[start] 现有 action: {len(out)} 条", flush=True)

    seen = set(x["user"] for x in out)
    t0 = time.time()

    for scene, tool in SCENES:
        text = call_api(scene, tool)
        if not text:
            time.sleep(1)
            continue
        samples = parse(text)
        for s in samples:
            if f'<call type="action"' not in s["assistant"]:
                continue
            if not (3 <= len(s["user"]) <= 80
                    and 10 <= len(s["assistant"]) <= 300):
                continue
            if s["user"] in seen:
                continue
            seen.add(s["user"])
            out.append({
                "type": "action",
                "user": s["user"],
                "assistant": s["assistant"],
            })
            print(f"[{len(out):>3}] {s['user'][:40]}", flush=True)

            # 每 20 条增量保存
            if len(out) % 20 == 0:
                # 合并回主文件
                merged = [x for x in existing if x.get("type") != "action"]
                merged += out
                with open(p, "w") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=2)
                print(f"  [save] 累计 action={len(out)}", flush=True)
        time.sleep(0.5)

    # 最终合并保存
    merged = [x for x in existing if x.get("type") != "action"]
    merged += out
    with open(p, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n[done] action={len(out)} 条 | {time.time()-t0:.0f}s")
    print(f"[save] {p}")

    from collections import Counter
    cnt = Counter(x["type"] for x in merged)
    print("=== 总分布 ===")
    for k, v in cnt.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
