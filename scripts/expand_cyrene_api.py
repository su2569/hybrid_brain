"""用 AMD Radeon Cloud (DeepSeek-V4-Flash) 扩充昔涟数据。"""
import json
import os
import random
import time
from openai import OpenAI

API_KEY = os.environ.get("RADEON_API_KEY", "")
if not API_KEY:
    raise SystemExit("请先 export RADEON_API_KEY=rc-xxx")

client = OpenAI(
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    api_key=API_KEY,
)

# ---------- 读原始数据做 few-shot ----------
SRC = "/mnt/workspace/data/cyrene_clean.json"
with open(SRC, encoding="utf-8") as f:
    raw = json.load(f)["samples"]

random.seed(42)
few_shot = random.sample(raw, min(8, len(raw)))
few_shot_text = "\n\n".join(
    f"用户：{s['input']}\n昔涟：{s['output']}" for s in few_shot
)

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
5. 输出格式严格为：
用户：xxx
昔涟：xxx

不要输出任何其他文字、解释或 markdown 标记。"""

# ---------- 场景 × 情绪矩阵 ----------
SCENES = [
    "用户第一次和昔涟打招呼",
    "用户在深夜睡不着",
    "用户分享今天遇到的开心事",
    "用户说自己很累",
    "用户问昔涟关于星星的事",
    "用户问昔涟的过去",
    "用户说要去远方旅行",
    "用户和昔涟道别",
    "用户在发呆",
    "用户问了一个哲学问题",
    "用户开玩笑逗昔涟",
    "用户送昔涟一个礼物",
    "用户说想听昔涟讲故事",
    "用户问昔涟喜欢什么季节",
    "用户说自己做了个梦",
    "用户问昔涟害怕什么",
    "用户分享了一首歌",
    "用户在雨天想起往事",
    "用户问昔涟相信命运吗",
    "用户说想和昔涟做约定",
    "用户问昔涟怎么看待告别",
    "用户说自己很孤独",
    "用户问昔涟有什么愿望",
    "用户说自己坚持不下去了",
    "用户问昔涟最喜欢什么颜色",
    "用户说想在星空下许愿",
    "用户问昔涟对时间的看法",
    "用户说自己犯了一个错误",
    "用户分享小时候的回忆",
    "用户问昔涟会唱歌吗",
    "用户说想和昔涟一起看日出",
    "用户问昔涟最珍惜什么",
    "用户分享了一首诗",
    "用户问昔涟喜欢雨天吗",
    "用户说想给昔涟画一幅画",
    "用户问昔涟对'命运'的理解",
    "用户说自己做了一个奇怪的梦",
    "用户问昔涟去过哪些地方",
    "用户和昔涟讨论记忆的意义",
    "用户问昔涟会哭吗",
    "用户说想成为昔涟的朋友",
    "用户问昔涟最喜欢哪颗星",
    "用户分享今天的晚餐",
    "用户问昔涟会不会忘记他",
    "用户说自己在远方想念昔涟",
    "用户问昔涟对'时间'的感受",
    "用户和昔涟聊起麦田的颜色",
    "用户问昔涟害怕孤独吗",
    "用户说想为昔涟摘一朵花",
    "用户问昔涟相信来世吗",
    "用户说自己最近在学画画",
    "用户问昔涟喜欢什么音乐",
    "用户说想听昔涟哼一首歌",
    "用户问昔涟如何面对失去",
    "用户分享今天的天气",
    "用户问昔涟会不会羡慕别人",
    "用户说自己刚刚哭过",
    "用户问昔涟有什么遗憾",
    "用户和昔涟聊起童年的夏天",
]

EMOTIONS = ["温柔", "俏皮", "怅然", "开心", "沉思", "关切",
            "平静", "怀念"]

# ---------- 生成 ----------
import re
from openai import APIConnectionError, APITimeoutError, RateLimitError

# ---------- 重试 + 限速 ----------
API_TIMEOUT = 60
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0      # 秒，指数退避基数
CALL_INTERVAL = 0.5         # 每次成功调用后 sleep 秒数

_last_call_ts = [0.0]


def _throttle():
    """保证调用间隔 >= CALL_INTERVAL"""
    now = time.time()
    delta = now - _last_call_ts[0]
    if delta < CALL_INTERVAL:
        time.sleep(CALL_INTERVAL - delta)
    _last_call_ts[0] = time.time()


def call_api(scene, emotion):
    """带重试和限速的 API 调用。返回文本或 None。"""
    user_msg = (
        f"场景：{scene}\n"
        f"情绪基调：{emotion}\n"
        f"请生成 1 组新的对话（用户提问 + 昔涟回答）。"
    )
    last_err = None

    for attempt in range(MAX_RETRIES):
        _throttle()
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
                timeout=API_TIMEOUT,
            )
            return resp.choices[0].message.content

        except RateLimitError as e:
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"[rate-limit] 第 {attempt+1} 次，等 {wait:.0f}s 重试")
            time.sleep(wait)
            last_err = e

        except (APITimeoutError, APIConnectionError) as e:
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"[network] {type(e).__name__}，等 {wait:.0f}s 重试")
            time.sleep(wait)
            last_err = e

        except Exception as e:
            msg = str(e)
            # 5xx 重试，4xx（除 429）直接放弃
            if "500" in msg or "502" in msg or "503" in msg or "504" in msg:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"[server-5xx] 等 {wait:.0f}s 重试")
                time.sleep(wait)
                last_err = e
            else:
                print(f"[give-up] {msg[:100]}")
                return None

    print(f"[retry-exhausted] {str(last_err)[:80]}")
    return None


# ---------- 更严格的解析和过滤 ----------
def parse(text):
    """从生成文本中提取 (user, assistant)。任何异常返回 None。"""
    if not text or not isinstance(text, str):
        return None

    # 去掉 markdown 代码块包裹
    text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    text = text.strip()

    user, assistant = None, None
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("用户："):
            user = line.split("：", 1)[-1].strip()
        elif line.startswith("昔涟："):
            assistant = line.split("：", 1)[-1].strip()

    if not user or not assistant:
        return None

    # 过滤太短/太长
    if not (5 <= len(user) <= 50):
        return None
    if not (20 <= len(assistant) <= 150):
        return None

    # 过滤包含明显异常字符的回答
    if any(kw in assistant for kw in ["用户", "系统", "Assistant", "User",
                                       "{}", "[]", "http"]):
        return None

    # 过滤重复字符（如"呀呀呀呀"）
    if re.search(r"(.)\1{5,}", assistant):
        return None

    return {"input": user, "output": assistant}


# ---------- 主流程 ----------
def main():
    target = 400
    out = []
    seen_inputs = set()

    stats = {"calls": 0, "fails": 0, "parse_fail": 0, "dup": 0}
    t0 = time.time()

    for scene in SCENES:
        if len(out) >= target:
            break
        emos = random.sample(EMOTIONS, 4)
        for emo in emos:
            if len(out) >= target:
                break
            for _ in range(2):
                if len(out) >= target:
                    break

                text = call_api(scene, emo)
                stats["calls"] += 1

                if text is None:
                    stats["fails"] += 1
                    continue

                d = parse(text)
                if d is None:
                    stats["parse_fail"] += 1
                    continue

                if d["input"] in seen_inputs:
                    stats["dup"] += 1
                    continue

                seen_inputs.add(d["input"])
                out.append(d)
                print(f"[{len(out):>3}] {emo} | {d['input'][:40]}")

    elapsed = time.time() - t0
    print(f"\n[done] {len(out)} 条 | calls={stats['calls']} "
          f"| fails={stats['fails']} | parse_fail={stats['parse_fail']} "
          f"| dup={stats['dup']} | elapsed={elapsed:.0f}s "
          f"| avg={elapsed/max(stats['calls'],1):.2f}s/call")

    out_path = "/mnt/workspace/data/cyrene_expanded.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"samples": out}, f, ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
