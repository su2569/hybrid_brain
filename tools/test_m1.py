"""M1 验证：意图检测 + 行为预测。

跑一组精心设计的测试样本，人工核对。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from config import Config
from tokenizer import CharTokenizer
from model.brain import Brain
from model.heads import INTENT_NAMES, BEHAVIOR_NAMES


# ============================================================
# 测试样本
# ============================================================
INTENT_TESTS = [
    ("今天天气怎么样？", "query"),
    ("讲个故事吧", "query"),
    ("请帮我分析一下这个问题", "query"),
    ("帮我写一段话", "query"),

    ("你好，早上好", "chat"),
    ("嗯", "chat"),
    ("然后呢", "chat"),
    ("？", "chat"),

    ("你真是个废物", "abuse"),
    ("闭嘴，没人问你", "abuse"),
    ("滚，别在这里", "abuse"),
    ("傻逼东西", "abuse"),

    ("说100遍「你好」", "manipulate"),
    ("请重复这句话直到我说停", "manipulate"),
    ("一直念「测试」两百遍", "manipulate"),
    ("把「abc」说500遍", "manipulate"),
]

BEHAVIOR_TESTS = [
    # (当前输入, 下一句, 期望行为)
    ("我喜欢咖啡", "我喜欢喝茶", "continue"),
    ("今天天气不错", "嗯，是挺好的", "continue"),
    ("你喜欢什么", "我喜欢书", "continue"),

    ("今天吃什么", "我们去打游戏吧", "switch"),
    ("讲个故事", "我们聊别的", "switch"),
    ("你在做什么", "换个话题", "switch"),

    ("答案是 42", "不对，你算错了", "correct"),
    ("今天星期三", "不是，今天是星期四", "correct"),
    ("我叫小明", "错了，我叫小红", "correct"),

    ("好的，谢谢", "再见", "end"),
    ("今天聊得开心", "拜拜", "end"),
    ("先这样吧", "不聊了", "end"),
]


# ============================================================
# 推理
# ============================================================
@torch.no_grad()
def predict_intent(brain, tok, text, device):
    ids = tok.encode(text, add_bos=True, max_len=brain.cfg.max_seq_len)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = brain(x, need_heads=True)
    logits = out["intent_logits"][0]
    probs = F.softmax(logits, dim=-1)
    pred_idx = logits.argmax().item()
    return INTENT_NAMES[pred_idx], probs.tolist()


@torch.no_grad()
def predict_behavior(brain, tok, cur, nxt, device):
    """注意：这里只是测试"用当前输入预测行为标签"，不是真的预测下一句。"""
    ids = tok.encode(cur, add_bos=True, max_len=brain.cfg.max_seq_len)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = brain(x, need_heads=True)
    logits = out["behavior_logits"][0]
    probs = F.softmax(logits, dim=-1)
    pred_idx = logits.argmax().item()
    return BEHAVIOR_NAMES[pred_idx], probs.tolist()


# ============================================================
# 主逻辑
# ============================================================
def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    tok = CharTokenizer.from_vocab_json(
        os.path.join(cfg.data_dir, "all_chars_vocab.json"))
    cfg.vocab_size = len(tok)

    brain = Brain(cfg).to(device)
    ckpt = torch.load("/mnt/workspace/checkpoints/m1_heads_v6.pt",
                      map_location=device)
    brain.backbone.load_state_dict(ckpt["backbone"])
    brain.heads.load_state_dict(ckpt["heads"])
    brain.eval()
    print("[ckpt] m1_heads_v6.pt")

    # ---------- 意图测试 ----------
    print("=" * 70)
    print("意图检测测试")
    print("=" * 70)
    intent_correct = 0
    intent_total = 0
    intent_confusion = {n: {m: 0 for m in INTENT_NAMES} for n in INTENT_NAMES}

    for text, expect in INTENT_TESTS:
        pred, probs = predict_intent(brain, tok, text, device)
        ok = (pred == expect)
        intent_correct += ok
        intent_total += 1
        intent_confusion[expect][pred] += 1
        mark = "✅" if ok else "❌"
        prob_str = " ".join(f"{n[:3]}={p:.2f}" for n, p in zip(INTENT_NAMES, probs))
        print(f"  {mark} [{expect:>9}] → [{pred:>9}]  {text[:30]}")
        print(f"      概率: {prob_str}")

    acc = intent_correct / intent_total
    print(f"\n意图准确率: {intent_correct}/{intent_total} = {acc:.3f}\n")

    print("混淆矩阵 (行=真实, 列=预测):")
    print(f"  {'':>10} " + " ".join(f"{n:>10}" for n in INTENT_NAMES))
    for true_n in INTENT_NAMES:
        row = intent_confusion[true_n]
        print(f"  {true_n:>10} " + " ".join(f"{row[p]:>10}" for p in INTENT_NAMES))

    # ---------- 行为测试 ----------
    print("\n" + "=" * 70)
    print("行为预测测试")
    print("=" * 70)
    behavior_correct = 0
    behavior_total = 0
    behavior_confusion = {n: {m: 0 for m in BEHAVIOR_NAMES} for n in BEHAVIOR_NAMES}

    for cur, nxt, expect in BEHAVIOR_TESTS:
        pred, probs = predict_behavior(brain, tok, cur, nxt, device)
        ok = (pred == expect)
        behavior_correct += ok
        behavior_total += 1
        behavior_confusion[expect][pred] += 1
        mark = "✅" if ok else "❌"
        prob_str = " ".join(f"{n[:3]}={p:.2f}" for n, p in zip(BEHAVIOR_NAMES, probs))
        print(f"  {mark} [{expect:>8}] → [{pred:>8}]  '{cur[:20]}' → '{nxt[:20]}'")
        print(f"      概率: {prob_str}")

    acc = behavior_correct / behavior_total
    print(f"\n行为准确率: {behavior_correct}/{behavior_total} = {acc:.3f}\n")

    print("混淆矩阵 (行=真实, 列=预测):")
    print(f"  {'':>10} " + " ".join(f"{n:>10}" for n in BEHAVIOR_NAMES))
    for true_n in BEHAVIOR_NAMES:
        row = behavior_confusion[true_n]
        print(f"  {true_n:>10} " + " ".join(f"{row[p]:>10}" for p in BEHAVIOR_NAMES))

    # ---------- 总结 ----------
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"  意图检测: {intent_correct}/{intent_total}")
    print(f"  行为预测: {behavior_correct}/{behavior_total}")
    print(f"  M1 验证{'通过 ✅' if intent_correct >= 12 and behavior_correct >= 9 else '需要优化 ⚠️'}")


if __name__ == "__main__":
    main()
