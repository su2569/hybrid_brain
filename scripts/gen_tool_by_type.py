"""按类型顺序生成 tool LoRA 数据（v2）。

关键改进：
  1. 场景池扩大 + 每次随机抽
  2. 变体前缀打破重复
  3. 连续 2 轮无进展才退出
  4. 总轮数上限 15
"""
import os, json, time, re, random
from openai import OpenAI

API_KEY = os.environ.get("RADEON_API_KEY", "")
if not API_KEY:
    raise SystemExit("export RADEON_API_KEY=rc-xxx")

client = OpenAI(
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    api_key=API_KEY,
)

# 变体前缀（打破"同场景重复"）
VARIANTS = [
    "",
    "换个角度",
    "用更口语化的表达",
    "从另一种情境出发",
    "结合具体的用户身份",
    "用年轻人口吻",
    "用商务场合口吻",
    "加入一些情绪",
]

PROMPT_TEMPLATES = [
    "请生成 5 组全新的对话样本。",
    "请生成 5 组不重复的对话。",
    "请生成 5 组真实场景的对话。",
    "请生成 5 组多样化的对话。",
    "请生成 5 组新鲜的对话样本。",
]


TYPE_SPECS = [
    {
        "type": "clarify",
        "target": 150,
        "system": """你正在生成"追问澄清"训练数据。

格式严格为：
用户：xxx
助手：xxx <call type="clarify">追问内容</call>

【核心规则】
1. 语气中性（"请问"、"能告诉我"），不要用"呀/呢/♪"
2. 必须在末尾输出 <call type="clarify">
3. 追问内容 5-20 字
4. 用户的问题必须**信息不足**
5. 严格输出格式：
用户：xxx
助手：xxx <call type="clarify">xxx</call>
不要任何解释。""",
        "scenes": [
            "用户问某商品价格但没说型号", "用户问路线但没说起点终点",
            "用户问症状但没描述具体情况", "用户要推荐但没说偏好",
            "用户问时间但没说是哪天", "用户问问题但表达很模糊",
            "用户要翻译但没说是哪种语言", "用户要代码但没说是哪种语言",
            "用户问天气但没说城市", "用户要对比但没说对比哪些",
            "用户问政策但没说地区", "用户要摘要但没给原文",
            "用户问价格但没说数量", "用户问故障但没说设备型号",
            "用户要方案但没说约束条件", "用户问航班但没说日期",
            "用户问餐馆但没说位置", "用户问书籍但没说要类型",
            "用户问电影但没说偏好", "用户问学习方法但没说科目",
            "用户问药但没说症状", "用户问课程但没说基础",
            "用户问股票但没说代码", "用户问合同但没说行业",
            "用户问签证但没说出国目的", "用户问贷款但没说金额",
            "用户问维修但没说故障", "用户问保险但没说险种",
            "用户问搬家但没说距离", "用户问装修但没说面积",
            "用户问搬家但没说时间", "用户问玩具但没说年龄段",
            "用户问化妆品但没说肤质", "用户问运动但没说基础",
            "用户问理财但没说风险", "用户问教育但没说年级",
            "用户问菜谱但没说食材", "用户问乐器但没说水平",
            "用户问花但没说场合", "用户问手机但没说预算",
            "用户问电脑但没说用途", "用户问相机但没说题材",
            "用户问书籍但没说要风格", "用户问旅行但没说季节",
            "用户问减肥但没说方法", "用户问健身但没说目标",
            "用户问兼职但没说时间", "用户问工作但没说行业",
            "用户问游戏但没说平台", "用户问动漫但没说要类型",
        ],
        "must_contain": '<call type="clarify"',
    },
    {
        "type": "suggest",
        "target": 100,
        "system": """你正在生成"主动建议"训练数据。

格式严格为：
用户：xxx
助手：xxx <call type="suggest">建议内容</call>

【核心规则】
1. 语气中性（"你可以"、"建议"）
2. 必须在末尾输出 <call type="suggest">
3. **建议内容 ≤ 15 字**（超短！）
4. 用户表达某种状态，你主动建议
5. 严格输出格式：
用户：xxx
助手：xxx <call type="suggest">短建议</call>
不要任何解释。""",
        "scenes": [
            "用户说今天很累", "用户说开心", "用户说在写作业",
            "用户说要去旅游", "用户说想学习", "用户说很无聊",
            "用户说要买东西", "用户说生病了", "用户说赶时间",
            "用户说想做菜", "用户说失眠", "用户说压力大",
            "用户说心情不好", "用户说想放松", "用户说想运动",
            "用户说想读书", "用户说肚子饿", "用户说工作忙",
            "用户说想存钱", "用户说想旅游", "用户说想画画",
            "用户说想唱歌", "用户说想学吉他", "用户说想健身",
            "用户说想早睡", "用户说想戒烟", "用户说想戒酒",
            "用户说想换工作", "用户说想搬家", "用户说想分手",
            "用户说想复合", "用户说想考试", "用户说想考证",
            "用户说想学英语", "用户说想学日语", "用户说想学编程",
            "用户说想开网店", "用户说想创业", "用户说想买房",
            "用户说想买车", "用户说想养宠物", "用户说想减肥",
            "用户说想脱单", "用户说想结婚", "用户说想生娃",
            "用户说想回家", "用户说想妈妈", "用户说想朋友",
            "用户说想哭", "用户说想笑", "用户说想睡",
        ],
        "must_contain": '<call type="suggest"',
    },
    {
        "type": "negative",
        "target": 150,
        "system": """你正在生成"普通对话"训练数据。

格式严格为：
用户：xxx
助手：xxx

【核心规则】
1. 助手回复**中性、简洁**（"好的"、"我明白了"、"这样啊"）
2. **绝对不能出现 <call 标签**
3. 不能追问、不能建议、不能调用工具
4. 回复长度 5-30 字
5. 严格输出格式：
用户：xxx
助手：xxx
不要任何解释。""",
        "scenes": [
            "用户打招呼", "用户说谢谢", "用户说再见", "用户说好的",
            "用户说嗯", "用户说收到", "用户说知道了", "用户说好的谢谢",
            "用户分享心情好", "用户分享心情不好", "用户说今天不错",
            "用户说天气好", "用户说吃过了", "用户说在忙", "用户说等一下",
            "用户说继续", "用户说然后呢", "用户说不错", "用户说有意思",
            "用户说真的吗", "用户说太好了", "用户说可惜", "用户说算了",
            "用户说没事", "用户说没关系", "用户说辛苦了", "用户说晚安",
            "用户说早安", "用户问几点了", "用户说在干嘛", "用户说吃饭了",
            "用户说下班了", "用户说出门了", "用户说回家了", "用户说睡了",
            "用户说醒了", "用户说有点困", "用户说有点饿", "用户说有点渴",
            "用户说有点热", "用户说有点冷", "用户说有点烦",
            "用户说知道了谢谢", "用户说好", "用户说行", "用户说OK",
            "用户说明白", "用户说收到啦", "用户说不错哦", "用户说嗯嗯",
        ],
        "must_contain": None,
        "must_not_contain": "<call",
    },
]


def call_api(scene, system, retries=5):
    variant = random.choice(VARIANTS)
    template = random.choice(PROMPT_TEMPLATES)
    user_msg = f"场景：{scene}\n"
    if variant:
        user_msg += f"要求：{variant}\n"
    user_msg += template

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="DeepSeek-V4-Flash",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=1.15, top_p=0.95, max_tokens=800, timeout=60,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "403" in err or "429" in err:
                wait = 5 * (2 ** attempt)
                print(f"    [限速 {attempt+1}/{retries}] 等 {wait}s",
                      flush=True)
                time.sleep(wait)
            else:
                print(f"    [retry {attempt+1}] {err[:60]}", flush=True)
                time.sleep(2)
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


def is_valid(sample, spec):
    a = sample["assistant"]
    u = sample["user"]
    if not (2 <= len(u) <= 80):
        return False
    if not (5 <= len(a) <= 300):
        return False
    if spec.get("must_contain") and spec["must_contain"] not in a:
        return False
    if spec.get("must_not_contain") and spec["must_not_contain"] in a:
        return False
    BAD = ["你是「昔涟」", "身份锚定", "性格核心", "句式禁令",
           "语气规则", "风格示例", "生成规则", "作为语言模型"]
    if any(b in a for b in BAD):
        return False
    return True


def main():
    path = "/mnt/workspace/data/tool_lora/raw.json"

    existing = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    print(f"[load] 现有 {len(existing)} 条", flush=True)

    from collections import Counter
    cnt = Counter(x.get("type") for x in existing)
    print(f"[stats] {dict(cnt)}", flush=True)

    t0 = time.time()
    MAX_ROUNDS = 15

    for spec in TYPE_SPECS:
        ctype = spec["type"]
        target = spec["target"]

        current = [x for x in existing if x.get("type") == ctype]
        print(f"\n{'='*60}", flush=True)
        print(f"[type] {ctype}  现有 {len(current)} / 目标 {target}",
              flush=True)
        print(f"{'='*60}", flush=True)

        if len(current) >= target:
            print(f"  [skip] 已达标", flush=True)
            continue

        seen = set(x["user"] for x in existing)
        consecutive_no_progress = 0

        for round_num in range(1, MAX_ROUNDS + 1):
            if len(current) >= target:
                break

            print(f"  [round {round_num}/{MAX_ROUNDS}] "
                  f"现有 {len(current)}/{target}", flush=True)

            before = len(current)

            # 每轮随机打乱场景顺序
            scenes = spec["scenes"][:]
            random.shuffle(scenes)

            for scene in scenes:
                if len(current) >= target:
                    break

                text = call_api(scene, spec["system"])
                if not text:
                    continue

                for s in parse(text):
                    if len(current) >= target:
                        break
                    if s["user"] in seen:
                        continue
                    if not is_valid(s, spec):
                        continue

                    seen.add(s["user"])
                    item = {"type": ctype, **s}
                    existing.append(item)
                    current.append(item)
                    print(f"    [{len(current):>3}] "
                          f"{s['user'][:30]}", flush=True)

                    if len(current) % 30 == 0:
                        with open(path, "w") as f:
                            json.dump(existing, f,
                                      ensure_ascii=False, indent=2)
                        print(f"      [save] 累计 {len(existing)}",
                              flush=True)

                time.sleep(0.8)  # 场景间隔

            # 本轮产出判断
            got = len(current) - before
            if got == 0:
                consecutive_no_progress += 1
                print(f"  [warn] 第 {round_num} 轮 0 产出，"
                      f"连续 {consecutive_no_progress} 轮", flush=True)
                if consecutive_no_progress >= 2:
                    print(f"  [exit] 连续 2 轮无进展，退出", flush=True)
                    break
            else:
                consecutive_no_progress = 0
                print(f"  [progress] 本轮 +{got}", flush=True)

        print(f"  [done] {ctype}: {len(current)} 条", flush=True)

    with open(path, "w") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"[ALL DONE] {len(existing)} 条 | "
          f"{time.time()-t0:.0f}s", flush=True)
    print(f"{'='*60}", flush=True)

    cnt = Counter(x.get("type") for x in existing)
    targets = {s["type"]: s["target"] for s in TYPE_SPECS}
    for k, v in cnt.items():
        t = targets.get(k, 0)
        status = "✅" if v >= t else "⚠️"
        print(f"  {status} {k}: {v} / {t}", flush=True)

    print(f"\n[save] {path}", flush=True)


if __name__ == "__main__":
    main()
