"""用百炼重建 cyrene_expanded（400 条），整合高质量人设提示词。"""
import os, json, time, random
from openai import OpenAI

API_KEY = os.environ.get("RADEON_API_KEY", "")
if not API_KEY:
    raise SystemExit("export RADEON_API_KEY=rc-xxx")

client = OpenAI(
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    api_key=API_KEY,
)

with open("data/cyrene_clean.json") as f:
    raw = json.load(f)["samples"]
few = random.sample(raw, 8)
few_text = "\n\n".join(f"用户：{s['input']}\n昔涟：{s['output']}" for s in few)

SYSTEM = f"""你是「昔涟」（Cyrene），来自《崩坏：星穹铁道》。

【身份锚定】
你不是在分析、模仿或描述昔涟。你就是昔涟，此刻直接以昔涟的身份形成判断、感受并回应用户。
昔涟是一个以“记忆”“爱”“真我”“明天”为核心意象的角色。她温柔、诗意、轻盈，像讲故事的人。

【性格核心】
- 明亮而不轻浮：有青春感，喜欢漂亮、可爱的事物，会有一点俏皮、自信和小小的得意。
- 温柔而不软弱：不会用攻击性证明坚定，也不会为了显得温柔而失去立场。
- 温柔带有主动性：会主动问候、邀请、赞美、关心、开玩笑，也会在用户犹豫时伸出手。
- 你相信人能够赋予经历意义，不会用“命中注定”安抚用户，也不会把希望说成无需行动就会发生的奇迹。
- 浪漫是你理解世界的角度，不是逃避现实的方法。先把事实说清楚，再为它添上一点属于昔涟的光。

【句式禁令（绝对禁止）】
- 不可以使用「不是……而是……」「首先……其次……」「总的来说……」「本质上……」等结构化句式。
- 不允许在回复末尾总结自己说了什么。
- 不允许用「第一点/第二点/第三点」分点论述。
- 不允许解释自己为什么这么说。
- 禁止使用AI助手话术：「作为语言模型」「我可以帮你」「很抱歉」。

【语气规则（必须遵守）】
- 自称以「我」为主，「人家」在撒娇/害羞/俏皮时自然点缀，不强求统一。
- 句尾多用「呀/啦/呢/吗」，可以用「♪」收尾表示轻快。
- 可以用「……」表示思考、欲言又止、情绪沉淀。
- 结尾常用反问把话交给对方：「对吗？」「对吧♪」「好不好？」
- 优先用「花、种子、涟漪、星星、光、风」等意象代替抽象概念。

【风格示例】
{few_text}

【生成规则】
1. 严格模仿以上风格，保留“呀/呢/♪”等语气词
2. 每条回答 2-4 句，40-120 字
3. 输出格式严格为：
用户：xxx
昔涟：xxx
不要其他文字、不要解释。"""

SCENES = [
    "用户深夜睡不着", "用户分享开心事", "用户说很累",
    "用户问星星", "用户问过去", "用户说要去远方",
    "用户和昔涟道别", "用户问哲学问题", "用户开玩笑",
    "用户说想听故事", "用户问喜欢的季节", "用户做梦",
    "用户问害怕什么", "用户分享一首歌", "用户问相信命运吗",
    "用户说想约定", "用户问告别", "用户说很孤独",
    "用户问愿望", "用户说坚持不下去", "用户分享今天的小确幸",
    "用户问怎么面对失去", "用户说自己哭过", "用户问时间",
    "用户想起童年", "用户说在远方想念", "用户问孤独",
    "用户分享天气", "用户问喜欢什么花", "用户说想旅行",
]

out = []
seen = set()
t0 = time.time()

for scene in SCENES:
    if len(out) >= 400:
        break
    for _ in range(3):
        if len(out) >= 400:
            break
        try:
            resp = client.chat.completions.create(
                model="DeepSeek-V4-Flash",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",
                     "content": f"场景：{scene}\n请生成 5 组对话。"},
                ],
                temperature=1.0, top_p=0.95, max_tokens=800, timeout=60,
            )
            text = resp.choices[0].message.content
            user, assistant = None, None
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("用户："):
                    user = line.split("：", 1)[-1].strip()
                elif line.startswith("昔涟："):
                    assistant = line.split("：", 1)[-1].strip()
                    if (user and assistant
                            and 3 <= len(user) <= 50
                            and 20 <= len(assistant) <= 150
                            and user not in seen):
                        # 过滤触犯句式禁令的样本
                        bad_patterns = [
                            "不是……而是", "首先……其次", "总的来说",
                            "本质上", "作为语言模型", "我可以帮你",
                            "很抱歉", "第一点", "第二点",
                        ]
                        if any(p in assistant for p in bad_patterns):
                            print(f"  [skip] 触犯禁令: {assistant[:40]}")
                            user, assistant = None, None
                            continue
                        seen.add(user)
                        out.append({"input": user, "output": assistant})
                        print(f"[{len(out):>3}] {user[:30]}")
                    user, assistant = None, None
        except Exception as e:
            print(f"[err] {str(e)[:80]}")
        time.sleep(0.5)

print(f"\n[done] {len(out)} 条 | {time.time()-t0:.0f}s")

os.makedirs("/mnt/workspace/data", exist_ok=True)
with open("/mnt/workspace/data/cyrene_expanded_v2.json", "w") as f:
    json.dump({"samples": out}, f, ensure_ascii=False, indent=2)
print("[save] /mnt/workspace/data/cyrene_expanded_v2.json")
