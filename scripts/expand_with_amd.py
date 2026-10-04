"""DeepSeek-V4-Flash 大批量扩充 cyrene 数据。"""
import os, json, random, time
from openai import OpenAI

client = OpenAI(
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    api_key=os.environ["RADEON_API_KEY"],
)
MODEL = "DeepSeek-V4-Flash"
OUT   = "data/cyrene_lora_v2/expanded_amd.json"

v2 = json.load(open("/mnt/workspace/data/cyrene_expanded_v2.json",
                    encoding="utf-8"))["samples"]
random.seed(42)
fewshot = random.sample(v2, 8)
EXAMPLES = "\n".join(f"用户：{x['input']}\n昔涟：{x['output']}" for x in fewshot)

def build_prompt(scene):
    return f"""你是昔涟，来自翁法罗斯的少女。性格温柔、喜欢听故事和看星星，语气轻盈。
严格模仿下面示例的语气，为新的用户输入生成回复。

【风格示例】
{EXAMPLES}

【硬性规则】
1. 回复长度 30-70 字，1-3 句
2. 自称「人家」或省略主语
3. 意象从这些里选 1-2 个：星星、麦田、涟漪、种子、风、记忆、光、露珠、黄昏
4. 不要堆砌意象（同一条最多用 2 个）
5. 禁止说教（「你可以试试」「建议你」「首先要」），禁止清单式回答
6. 结尾最多 30% 用 ♪
7. 不要直接给解决方案，先接住情绪

【新输入】
用户：{scene}
昔涟："""

SCENES = [
    # 技术/学习
    "代码跑不通，debug 了好久都没找到问题",
    "论文看不懂，公式太多了",
    "学新语言好难，记不住单词",
    "考试考砸了，感觉白复习了",
    "写论文卡住了，一个字也憋不出来",
    "项目代码被别人改乱了",
    "刚学的知识第二天就忘了",
    "想自学但坚持不了几天",
    "面试被问到不会的问题",
    "看教程看着看着就走神了",
    # 工作
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
    # 人际
    "和朋友吵架了，现在很尴尬",
    "被家人误解了，解释不清楚",
    "喜欢的人不回消息",
    "朋友借钱不还，不好意思开口",
    "室友太吵，睡不好",
    "感觉被朋友孤立了",
    "被相亲对象拒绝了",
    "很久没联系的朋友突然找我",
    "同事总在背后说我坏话",
    "妈妈总是催我结婚",
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
    # 节日/纪念
    "今天生日，但没人记得",
    "和恋人分手一周年了",
    "明天是妈妈的忌日",
    "除夕一个人过",
    "中秋节一个人看月亮",
    # 食物/日常
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
    "楼下的流浪猫今天不见了",
    "朋友把猫寄养在我家，有点手忙脚乱",
    "路边看到一只受伤的小鸟",
    # 金钱
    "这个月花超了，月底要吃土",
    "想买的东西好贵，舍不得",
    "理财亏了，好心疼",
    "被亲戚借钱，不知道怎么拒绝",
    "年终奖比预期少很多",
    "刚发工资就还完信用卡了",
    "看到别人晒房晒车好焦虑",
    # 情绪低谷
    "突然很想哭，不知道为什么",
    "什么都不想做，只想躺着",
    "感觉被困住了，走不出去",
    "对未来完全没希望",
    "觉得活着好累",
    "讨厌现在的自己又改变不了",
    # 小确幸
    "今天看到一只特别可爱的小狗",
    "刚喝到一杯超好喝的奶茶",
    "路上捡到一张明信片",
    "今天天气特别好，阳光很暖",
    "刚做完一件拖了很久的事",
    "窗外飘来一阵桂花香",
    # 情感
    "暗恋的人有对象了",
    "和前任在街上碰到了",
    "异地恋好累，快坚持不住了",
    "对方父母不同意我们在一起",
    "刚失恋，心好痛",
    "结婚三年，越来越没话说",
    # 深夜
    "凌晨三点，一个人醒着",
    "窗外的雨声让人睡不着",
    "深夜突然想起以前的事",
    "半夜做了噩梦，心跳好快",
    "一个人吃夜宵有点孤单",
]

VARIANTS = 3
random.seed(7)

results = []
total = len(SCENES) * VARIANTS
i = 0

for scene in SCENES:
    for v in range(VARIANTS):
        i += 1
        prompt = build_prompt(scene)
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=200,
            )
            reply = resp.choices[0].message.content.strip()
            reply = reply.replace("昔涟：", "").strip()
            if len(reply) < 15 or len(reply) > 200:
                print(f"[{i}/{total}] SKIP len={len(reply)}")
                continue
            results.append({"input": scene, "output": reply})
            print(f"[{i}/{total}] {scene[:20]} → {reply[:35]}...")
        except Exception as e:
            print(f"[{i}/{total}] FAIL {str(e)[:60]}")
        time.sleep(0.3)

json.dump({"samples": results}, open(OUT, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n[ok] {len(results)} 条 → {OUT}")
