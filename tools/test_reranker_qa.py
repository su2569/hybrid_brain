"""验证 reranker 在 QA 数据上的绝对排序能力。"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from config import Config
from tokenizer import CharTokenizer
from model.brain import Brain
from rag.kb_builder import load_dureader


def main():
    cfg = Config()
    device = torch.device("cuda")
    tok = CharTokenizer.from_vocab_json("data/all_chars_vocab.json")
    cfg.vocab_size = len(tok)

    brain = Brain(cfg).to(device)
    ckpt = torch.load("/mnt/workspace/checkpoints/final.pt", map_location=device)
    state = {k.replace("backbone.", ""): v for k, v in ckpt["model"].items()
             if k.startswith("backbone.")}
    brain.backbone.load_state_dict(state if state else ckpt["model"], strict=False)

    rckpt = torch.load("/mnt/workspace/checkpoints/qa_reranker.pt", map_location=device)
    brain.retriever.reranker.load_state_dict(rckpt["reranker"])
    brain.eval()
    brain.retriever.reranker.eval()

    # 加载 QA 数据
    samples = load_dureader(max_n=20000)
    print(f"[data] 全部 {len(samples)} 条")
    # 训练用了前 10000 条，测试从后面取
    if len(samples) > 10000:
        test = samples[10000:10500]
    else:
        # 数据不够，取最后 10%
        n_test = max(100, len(samples) // 10)
        test = samples[-n_test:]
    print(f"[test] {len(test)} 条")

    top1_correct = 0
    top5_correct = 0
    mrr_sum = 0.0

    with torch.no_grad():
        for i, s in enumerate(test):
            q = s["question"]
            gt = s["passage"]

            # 构造候选：1 正 + 9 随机负（从训练集部分采样）
            neg_pool = samples[:10000] if len(samples) > 10000 else samples
            # 排除和 gt 相同的
            neg_pool = [n for n in neg_pool if n["passage"] != gt]
            negs = random.sample(neg_pool, min(9, len(neg_pool)))
            cands = [gt] + [n["passage"] for n in negs]

            # 编码
            q_ids = tok.encode(q, add_bos=True, max_len=cfg.max_seq_len)
            q_x = torch.tensor([q_ids], dtype=torch.long, device=device)
            q_h = brain.backbone(q_x)["hidden"]

            cand_hiddens = []
            for c in cands:
                ids = tok.encode(c, add_bos=True, max_len=cfg.max_seq_len)
                x = torch.tensor([ids], dtype=torch.long, device=device)
                h = brain.backbone(x)["hidden"][0]  # [T, d]
                cand_hiddens.append(h)

            L = max(h.size(0) for h in cand_hiddens)
            padded = torch.zeros(len(cand_hiddens), L, cfg.d_model, device=device)
            for j, h in enumerate(cand_hiddens):
                padded[j, :h.size(0)] = h

            scores = brain.retriever.reranker(q_h, padded)  # [10]
            _, idx = scores.topk(10)
            rank_t = (idx == 0).nonzero().flatten()
            rank = int(rank_t[0].item()) if rank_t.numel() > 0 else 999

            if rank == 0:
                top1_correct += 1
            if rank < 5:
                top5_correct += 1
            mrr_sum += 1.0 / (rank + 1)

    n = len(test)
    print(f"\n=== Reranker 绝对排序能力 ===")
    print(f"Top-1 命中率: {top1_correct/n:.3f}")
    print(f"Top-5 命中率: {top5_correct/n:.3f}")
    print(f"MRR: {mrr_sum/n:.3f}")
    print(f"\n对比：随机猜测 Top-1 = {1/10:.3f}")


if __name__ == "__main__":
    main()
