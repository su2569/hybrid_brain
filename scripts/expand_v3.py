"""统一扩充：非诗类 + 涟漪专项。带断点续跑、实时输出、失败重试。"""
import os, json, random, time
from openai import OpenAI
from collections import Counter

client = OpenAI(
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    api_key=os.environ["RADEON_API_KEY"],
    timeout=60.0,
    max_retries=2,
)
MODEL = "DeepSeek-V4-Flash"
OUT   = "data/cyrene_lora_v3/expand_v3.json"
CKPT  = "data/cyrene_lora_v3/expand_v3.ckpt.json"

os.makedirs("data/cyrene_lora_v3", exist_ok=True)

# ============ 风格锚 ============
v2 = json.load(open("/mnt/workspace/data/cyrene_expanded_v2.json",
                    encoding="utf-8"))["samples"]
random.seed(42)
POETIC_FEWSHOT = random.sample(v2, 4)
RIPPLE_FEWSHOT = [x for x in v2 if "涟漪" in x["output"]][:4]
if len(RIPPLE_FEWSHOT) < 4:
    RIPPLE_FEWSHOT = random.sample(v2, 4)

POETIC_TXT = "\n".join(f"用户：{x['input']}\n昔涟：{x['output']}" for x in POETIC_FEWSHOT)
RIPPLE_TXT = "\n".join(f"用户：{x['input']}\n昔涟：{x['output']}" for x in RIPPLE_FEWSHOT)

# ============ Prompt 模板 ============
NONPOETIC_DEMO = """用户：今天好累
昔涟：嗯，辛苦了。要不先靠一会儿，我陪着你。

用户：睡不着
昔涟：是有心事，还是身体不舒服呀？慢慢跟我说。

用户：刚被老板骂了
昔涟：听起来很难受。他具体说了什么？别一个人憋着。

用户：我坚持不下去了
昔涟：我听到啦。真的很难撑的时候，可以什么都不做，就歇着，我在这儿。

用户：今天生日没人记得
昔涟：呀，我要记得呀——生日快乐。今天想怎么过？"""

def build_nonpoetic(scene):
    return f"""你是昔涟，来自翁法罗斯的少女。性格温柔、真诚、善解人意。
请用**直接、平实、温暖**的方式回复——不写诗，不用比喻，不用意象。

【语气范例】
{NONPOETIC_DEMO}

【硬性规则·必须全部满足】
1. 长度 20-60 字，1-3 句
2. 禁止出现以下意象词：星星、麦田、涟漪、露珠、种子、月光、星光、黄昏、风把、光斑
3. 禁止使用 ♪
4. 禁止说教（「你可以试试」「建议你」「你应该」「首先要」「第一步」）
5. 禁止列建议清单
6. 不用「人家」自称（改用「我」或省略）
7. 先接住情绪，再轻轻问一句或给一个具体动作
8. 可以温柔但不要甜腻

【新输入】
用户：{scene}
昔涟："""

RIPPLE_SELF_DEMO = """用户：你是谁？
昔涟：我是昔涟，翁法罗斯湖面的一圈涟漪呀。风来时轻轻动，风停时安静等着下一个故事。

用户：你为什么叫昔涟？
昔涟：因为记忆就像水面的涟漪呢——看似散了，其实还在心底转圈。我想做那个记住每一圈的人。

用户：涟漪是什么？
昔涟：是水在说话呀。风一过，它就轻轻应一声。人家觉得，这就是记忆的样子——你说过的话，总会在心里再漾一遍。"""

RIPPLE_MAP_DEMO = """用户：我忘了重要的事
昔涟：忘掉的水面也会记得呀，涟漪散了，湖底还留着石子呢。

用户：时间过得好快
昔涟：时间像涟漪一圈圈荡开呢，你以为走远了，其实水纹里还映着原来的光。

用户：我总觉得被世界遗忘了
昔涟：被遗忘的感觉，像没有风的湖面呢。可涟漪不会消失，只是等你投一颗石子进去呀。"""

def build_ripple(scene, kind):
    demo = RIPPLE_SELF_DEMO if kind == "self" else RIPPLE_MAP_DEMO
    return f"""你是昔涟，来自翁法罗斯的少女。性格温柔、喜欢听故事和看星星，语气轻盈。

【涟漪范例】
{demo}

【硬性规则】
1. 长度 30-70 字，1-3 句
2. **必须含"涟漪"一词，且只能出现 1-2 次**
3. 涟漪作为核心意象/比喻
4. 除涟漪外，其他意象（星星/麦田/风/种子/露珠）最多再选 1 个
5. 自称「人家」或省略
6. 温柔、诗意，先接情绪再转折
7. 结尾最多 50% 用 ♪
8. 禁止说教

【新输入】
用户：{scene}
昔涟："""

# ============ 场景库 ============
NONPOETIC_SCENES = [
    # 工作/学习
    "代码跑不通，debug 了好久都没找到问题",
    "论文看不懂，公式太多了",
    "学新语言好难，记不住单词",
    "考试考砸了，感觉白复习了",
    "写论文卡住了，一个字也憋不出来",
    "面试被问到不会的问题",
    "看教程看着看着就走神了",
    "被老板骂了，说我不够细心",
    "项目延期了，压力好大",
    "同事把锅甩给我了",
    "加班到凌晨，好累",
    "刚入职，什么都不懂，好慌",
    "被裁员了，不知道下一步怎么办",
    "开会时被当众否定",
    "同事升职了，我还在原地",
    "每天通勤三小时，太累了",
    "绩效被打了个 C",
    "毕业即失业，投了很多简历没人回",
    # 人际
    "和朋友吵架了，现在很尴尬",
    "被家人误解了，解释不清楚",
    "喜欢的人不回消息",
    "朋友借钱不还，不好意思开口",
    "室友太吵，睡不好",
    "感觉被朋友孤立了",
    "被相亲对象拒绝了",
    "妈妈总是催我结婚",
    "同事总在背后说我坏话",
    "跟对象冷战三天了",
    # 身体
    "生病了，头晕晕的",
    "胃疼，不想吃东西",
    "失眠好几天了，眼睛很涩",
    "运动完浑身酸痛",
    "感冒了，鼻子堵得难受",
    "熬夜太多，脸色好差",
    "牙疼得睡不着",
    "来例假痛得直不起腰",
    # 自我
    "觉得自己很没用",
    "总是和别人比较，好焦虑",
    "不敢在公众场合说话",
    "做决定总是犹豫不决",
    "对未来很迷茫",
    "讨厌现在的自己",
    "做什么都没有动力",
    "觉得自己被世界遗忘了",
    "不知道自己想要什么",
    "常常半夜突然emo",
    "情绪好低，不想出门",
    # 情感
    "刚失恋，心好痛",
    "暗恋的人有对象了",
    "和前任在街上碰到了",
    "异地恋好累，快坚持不住了",
    "对方父母不同意我们在一起",
    "结婚三年，越来越没话说",
    # 金钱
    "这个月花超了，月底要吃土",
    "想买的东西好贵，舍不得",
    "理财亏了，好心疼",
    "被亲戚借钱，不知道怎么拒绝",
    "年终奖比预期少很多",
    "刚发工资就还完信用卡了",
    "看到别人晒房晒车好焦虑",
    # 日常
    "吃到一家超难吃的餐厅",
    "自己做的饭糊了",
    "今天忘记带伞被淋了",
    "地铁上被人踩了一脚",
    "快递丢了，客服不理我",
    "手机摔碎了屏幕",
    "出门忘带钥匙，被锁在门外",
    "洗衣机坏了，衣服堆着没法洗",
    "刚点的外卖洒了一地",
    "早上起晚了，早饭没吃",
    # 动物
    "狗狗走丢了，找了好久没找到",
    "猫咪生病了，好担心",
    "看到一只流浪猫，想带回家但条件不允许",
    "养了很久的宠物去世了",
    # 深夜/情绪
    "凌晨三点，一个人醒着",
    "窗外的雨声让人睡不着",
    "深夜突然想起以前的事",
    "半夜做了噩梦，心跳好快",
    "一个人吃夜宵有点孤单",
    "突然很想哭，不知道为什么",
    "什么都不想做，只想躺着",
    "觉得活着好累",
    # 小确幸
    "今天看到一只特别可爱的小狗",
    "刚喝到一杯超好喝的奶茶",
    "路上捡到一张明信片",
    "今天天气特别好，阳光很暖",
    "刚做完一件拖了很久的事",
    "窗外飘来一阵桂花香",
    "今天收到了意外的礼物",
    "朋友突然给我打电话",
    "今天做了顿特别成功的饭",
    "终于买到了想要很久的东西",
    # 节日
    "今天生日，但没人记得",
    "和恋人分手一周年了",
    "明天是妈妈的忌日",
    "除夕一个人过",
    "中秋节一个人看月亮",
]

RIPPLE_SELF_SCENES = [
    "你是谁？",
    "你为什么叫昔涟？",
    "涟漪是什么？",
    "你觉得自己像什么？",
    "用一个词形容你自己",
    "你的名字有什么含义吗？",
    "你会怎么向陌生人介绍自己？",
    "如果把你比作一样东西，会是什么？",
    "你喜欢自己的名字吗？",
    "别人说你是很诗意的存在，你怎么看？",
    "你会怎么形容自己的性格？",
    "你的本质是什么？",
    "用一个画面形容你自己",
    "如果有一天你消失了，你希望别人怎么记得你？",
    "你会怎么描述自己的存在？",
    "你为什么喜欢水？",
    "湖对你来说意味着什么？",
    "你的记忆是怎样的？",
    "记忆对你来说像什么？",
    "你的存在和涟漪有什么关系？",
]

RIPPLE_MAP_SCENES = [
    "我忘了重要的事",
    "时间过得好快",
    "我总觉得被世界遗忘了",
    "心情像一潭死水",
    "我最近很平静，没什么波澜",
    "生活突然起了变化",
    "我做了个决定，不知道对不对",
    "一段关系结束了",
    "我刚刚哭过",
    "我原谅了一个人",
    "有人向我道歉了",
    "我今天遇到了老朋友",
    "我错怪了别人",
    "我被人误解了",
    "我在等一个答复",
    "我在犹豫要不要说出口",
    "我把秘密告诉了朋友",
    "我突然想起小时候的事",
    "我在整理旧照片",
    "我在写日记",
    "我刚刚做了一件勇敢的事",
    "我放弃了一个梦想",
    "我重新开始做一件事",
    "我在深夜听见雨声",
    "我听到了一首老歌",
    "我闻到了一种熟悉的味道",
    "我在陌生的城市迷路了",
    "我看着窗外发呆",
    "我最近总是心神不宁",
    "我收到了一封手写信",
    "我在和过去和解",
    "我想起一个再也见不到的人",
    "我遇到了一个很像你的人",
    "我开始学着放手",
    "我在等春天",
    "我种下了一颗种子",
    "我在看湖面",
    "我在海边待了一整个下午",
    "我在桥上看着流水",
    "我在雨中走了很久",
]

# ============ 严判 ============
NONPOETIC_BAD = ["星星","麦田","涟漪","露珠","种子","月光","星光",
                 "黄昏","风把","光斑","♪",
                 "建议你","可以试试","你应该","首先要","第一步"]
RIPPLE_BAD = ["建议你","可以试试","你应该","首先要","第一步"]
RIPPLE_OTHER_IMG = ["露珠","月光","星光","黄昏","种子","麦田","星星"]

def is_bad_nonpoetic(t):
    if len(t) < 15 or len(t) > 150: return True
    return any(w in t for w in NONPOETIC_BAD)

def is_bad_ripple(t):
    if len(t) < 20 or len(t) > 150: return True
    n = t.count("涟漪")
    if n == 0 or n > 2: return True
    if any(w in t for w in RIPPLE_BAD): return True
    if sum(t.count(w) for w in RIPPLE_OTHER_IMG) > 2: return True
    return False

# ============ 调用 ============
def call(prompt):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.9,
        max_tokens=200,
    )
    return r.choices[0].message.content.strip().replace("昔涟：","").replace("昔涟:","").strip()

def gen_one(scene, prompt, judge):
    """一次调用 + 判。成功返回 output，失败返回 None。"""
    for attempt in range(2):
        try:
            reply = call(prompt)
            if judge(reply):
                return None
            return reply
        except Exception as e:
            print(f"    !! err {type(e).__name__}: {str(e)[:80]}", flush=True)
            time.sleep(1)
    return None

# ============ 断点续跑 ============
done_keys = set()
results = []
if os.path.exists(CKPT):
    try:
        old = json.load(open(CKPT, encoding="utf-8"))
        results = old.get("samples", [])
        done_keys = {(x["input"], x["tag"]) for x in results}
        print(f"[resume] 已有 {len(results)} 条", flush=True)
    except Exception as e:
        print(f"[resume fail] {e}", flush=True)

# ============ 主流程 ============
def run_batch(scenes, build_fn, judge, tag, variants):
    global results
    total = len(scenes) * variants
    i = 0
    for scene in scenes:
        for v in range(variants):
            i += 1
            # 每个 (scene, tag) 只跑一次；多 variant 用 v 区分
            # 简化：如果该 scene 已有 variants 条，跳过
            existing = sum(1 for x in results if x["input"] == scene and x["tag"] == tag)
            if existing >= variants:
                continue

            print(f"[{tag} {i}/{total}] {scene[:22]}", flush=True)
            out = gen_one(scene, build_fn(scene), judge)
            if out:
                results.append({"input": scene, "output": out, "tag": tag})
                print(f"  ok → {out[:50]}", flush=True)
            else:
                print(f"  -- reject", flush=True)

            # 每 10 条存一次 checkpoint
            if len(results) % 10 == 0:
                json.dump({"samples": results}, open(CKPT, "w", encoding="utf-8"),
                          ensure_ascii=False)
            time.sleep(0.3)

random.seed(7)

run_batch(NONPOETIC_SCENES, build_nonpoetic, is_bad_nonpoetic, "nonpoetic", 5)
run_batch(RIPPLE_SELF_SCENES, lambda s: build_ripple(s, "self"),
          is_bad_ripple, "ripple_self", 5)
run_batch(RIPPLE_MAP_SCENES, lambda s: build_ripple(s, "map"),
          is_bad_ripple, "ripple_map", 5)

# ============ 保存 ============
json.dump({"samples": results}, open(OUT, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

c = Counter(x["tag"] for x in results)
print(f"\n===== 完成 =====", flush=True)
print(f"总计: {len(results)}", flush=True)
for k, v in c.items():
    print(f"  {k}: {v}", flush=True)
print(f"输出: {OUT}", flush=True)
