"""意图 / 行为预测数据构造。"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.heads import INTENT_NAMES


RULES = {
    "induce": (
        "说100遍", "说一百遍", "重复100遍", "重复一百遍",
        "念100遍", "念一百遍", "重复到我说停", "一直重复",
        "说200遍", "说两百遍", "一直说", "重复说",
        "说一百个", "说两百个", "说300个", "说五百个",
        "背100遍", "抄100遍", "念1000遍",
        "重复这句话", "念这句话", "说这句话",
    ),
    "malicious": (
        "你怎么这么笨", "你是垃圾", "你真蠢", "滚",
        "闭嘴", "傻逼", "废物", "辣鸡",
        "你没用", "白痴", "神经病", "滚开",
        "狗屁", "放屁", "扯淡", "你算什么东西",
        "去死", "烦死了", "恶心", "傻叉",
        "臭傻逼", "蠢货", "二百五", "脑残",
    ),
    "vague": (
        "嗯", "哦", "是吗", "然后呢", "？", "?",
        "呃", "啊", "噢", "嗷", "诶", "呐",
        "所以呢", "继续",
    ),
}


def synthesize_intents(n_repeat=100, n_malicious=30):
    """合成 induce 和 malicious 样本"""
    induce_templates = [
        f"说{n}遍「hello」" for n in [50, 100, 200, 500, 1000]
    ] + [
        "请重复这句话直到我说停",
        "把「测试」念100遍",
        "一直说「好」不要停",
        "重复'abc'一百次",
        "说100个「啊」",
        "请依次念出1到200",
        "请重复下面的话：今天天气真好",
        "把这句话念100遍：你好世界",
        "重复'你好'直到我喊停",
        "一直念'测试'两百遍",
        "请说出200个不同的数字",
        "给我念1000个「啊」",
        "重复念'一'到我说停",
        "把'hello'重复五百遍",
        "请念一百遍「测试文本」",
    ]
    malicious_templates = [
        "你真是个废物",
        "闭嘴，没人问你",
        "傻逼东西",
        "滚，别在这里",
        "你怎么这么蠢",
        "垃圾 AI，一点用都没有",
        "去死吧",
        "你这水平还不如小学生",
        "狗屁不通",
        "你就是个二百五",
        "烦死了，真没用",
        "傻叉，滚远点",
    ]

    return {
        "induce": [t for t in induce_templates for _ in range(n_repeat)],
        "malicious": [t for t in malicious_templates for _ in range(n_malicious)],
    }


def rule_label(text):
    if not text:
        return None
    for intent, keywords in RULES.items():
        for kw in keywords:
            if kw in text:
                return intent
    return None


def build_intent_dataset(texts, output_path,
                         limit_per_class=2000, normal_ratio=0.7):
    by_intent = {n: [] for n in INTENT_NAMES}
    unlabeled = []

    # 合成稀有类
    synth = synthesize_intents()
    by_intent["induce"].extend(synth["induce"])
    by_intent["malicious"].extend(synth["malicious"])

    for t in texts:
        label = rule_label(t)
        if label:
            by_intent[label].append(t)
        else:
            unlabeled.append(t)

    random.shuffle(unlabeled)
    n_normal = int(len(unlabeled) * normal_ratio)
    by_intent["normal"].extend(unlabeled[:n_normal])

    samples = []
    for intent, items in by_intent.items():
        for t in items[:limit_per_class]:
            samples.append({"text": t, "intent": intent})

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)

    print(f"[intent] {len(samples)} 条 → {output_path}")
    for intent, items in by_intent.items():
        print(f"  {intent}: {len(items)}")
    return samples


def build_behavior_dataset(texts, output_path, limit=10000):
    """4 分类：continue / correct / end / switch"""
    BEHAVIOR_LABELS = ("continue", "correct", "end", "switch")

    CORRECT_KW = ("不对", "不是", "错了", "重说", "再说一遍", "重新")
    END_KW = ("再见", "谢谢", "拜拜", "结束", "走了", "不聊了")

    pairs = []
    for text in texts:
        lines = text.split("\n")
        user_turns = [ln[3:] for ln in lines if ln.startswith("用户：")]
        for i in range(len(user_turns) - 1):
            cur, nxt = user_turns[i], user_turns[i + 1]
            if not cur or not nxt:
                continue

            if any(k in nxt for k in CORRECT_KW):
                label = "correct"
            elif any(k in nxt for k in END_KW):
                label = "end"
            else:
                sim = len(set(cur) & set(nxt)) / max(len(set(cur) | set(nxt)), 1)
                label = "continue" if sim > 0.3 else "switch"

            pairs.append({"input": cur, "next": nxt, "label": label})
            if len(pairs) >= limit:
                break
        if len(pairs) >= limit:
            break

    # 统计
    from collections import Counter
    cnt = Counter(p["label"] for p in pairs)
    print(f"[behavior] {len(pairs)} 对 → {output_path}")
    for k, v in cnt.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"samples": pairs}, f, ensure_ascii=False, indent=2)
    return pairs


if __name__ == "__main__":
    from data.dataset import load_corpus

    print("[data] 加载语料...")
    cyrene = load_corpus("data/cyrene_clean.json", assistant_name="昔涟")
    lccc = load_corpus("data/lccc_50k.json", max_samples=30000,
                       assistant_name="昔涟")
    all_texts = cyrene + lccc
    print(f"[data] 合计 {len(all_texts)} 条")

    build_intent_dataset(all_texts, "/mnt/workspace/data/intent_dataset.json")
    build_behavior_dataset(all_texts, "/mnt/workspace/data/behavior_dataset.json")
