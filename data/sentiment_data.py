"""情感极性数据：positive / negative / neutral。"""
import json
import os
import random


POSITIVE = [
    "太好了", "真棒", "谢谢你", "辛苦了", "做得不错",
    "我很开心", "今天心情好", "运气不错", "喜欢这个",
    "你真厉害", "好厉害呀", "真有帮助", "完美",
    "爱了爱了", "绝了", "舒服", "满意", "开心",
    "这主意不错", "好主意", "太棒了", "赞", "给力",
    "谢谢你帮我", "你真聪明", "真贴心", "太感动了",
    "我很喜欢", "挺好", "不错不错", "非常好",
    "真的很好", "特别满意", "超赞", "无敌",
    "非常喜欢", "我太爱了", "太优秀了", "厉害",
    "好棒", "太好啦", "真妙", "好呀", "行呀",
    "给力啊", "有道理", "说得对", "就是这样", "确实",
]
POSITIVE = (POSITIVE * 8)[:300]

NEGATIVE = [
    "太难过了", "我很伤心", "真倒霉", "好烦", "讨厌",
    "不开心", "心情不好", "好累啊", "压力好大",
    "难受", "痛苦", "糟糕", "遗憾", "失望",
    "好气啊", "气死我了", "烦死了", "不想活了",
    "太难了", "真难", "好难", "绝望", "没希望",
    "我很焦虑", "担心", "害怕", "恐惧", "不安",
    "委屈", "孤单", "寂寞", "无助", "崩溃",
    "抑郁", "沮丧", "低落", "郁闷", "烦躁",
    "恶心", "讨厌死了", "恨", "怨", "苦",
    "倒霉透顶", "生无可恋", "没意思", "无聊", "够呛",
]
NEGATIVE = (NEGATIVE * 8)[:300]

NEUTRAL = [
    "今天几号", "现在几点", "什么天气", "几点了",
    "这是什么", "什么意思", "怎么做", "为什么",
    "在哪里", "多少", "有吗", "可以吗",
    "介绍一下", "解释一下", "说一下", "讲讲",
    "嗯", "哦", "好的", "知道了", "明白了",
    "然后呢", "所以呢", "继续", "接着",
    "在吗", "在不在", "有空吗", "忙吗",
    "你好", "早上好", "晚上好", "晚安",
    "是的", "不是", "可能", "也许", "大概",
    "行", "可以", "不行", "算了", "随便",
    "看情况", "再说", "看看吧", "考虑一下", "再想想",
]
NEUTRAL = (NEUTRAL * 8)[:300]


def build(output_path):
    samples = []
    for t in POSITIVE:
        samples.append({"text": t, "sentiment": "positive"})
    for t in NEGATIVE:
        samples.append({"text": t, "sentiment": "negative"})
    for t in NEUTRAL:
        samples.append({"text": t, "sentiment": "neutral"})

    random.shuffle(samples)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)

    from collections import Counter
    cnt = Counter(s["sentiment"] for s in samples)
    print(f"[sentiment] {len(samples)} 条 → {output_path}")
    for k, v in cnt.items():
        print(f"  {k}: {v}")
    return samples


if __name__ == "__main__":
    build("/mnt/workspace/data/sentiment_dataset.json")
