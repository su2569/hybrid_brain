"""Hard negative 测试：候选都从 cosine top-50 里取。v2（带 mask）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from config import Config
from tokenizer import CharTokenizer
from model.brain import Brain
from rag.kb_builder import load_dureader


def pad_with_mask(ids_list, max_len, device):
    """返回 padded [N, L] 和 mask [N, L]（True=有效）"""
    L = max(len(x) for x in ids_list)
    padded = torch.zeros(len(ids_list), L, dtype=torch.long, device=device)
    mask = torch.zeros(len(ids_list), L, dtype=torch.bool, device=device)
    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = torch.tensor(ids, device=device)
        mask[i, :len(ids)] = True
    return padded, mask


def main():
    cfg = Config()
    device = torch.device("cuda")
    tok = CharTokenizer.from_vocab_json("data/all_chars_vocab.json")
    cfg.vocab_size = len(tok)

    brain = Brain(cfg).to(device)
    ckpt = torch.load("/mnt/workspace/checkpoints/final.pt", map_location=device)
    state = {k.replace("backbone.", ""): v for k, v in ckpt["model"].items()
             if k.startswith("backbone.")}
    brain.backbone.load_state_dict(state if state else ckpt["model"],
                                   strict=False)
    rckpt = torch.load("/mnt/workspace/checkpoints/qa_reranker.pt",
                       map_location=device)
    brain.retriever.reranker.load_state_dict(rckpt["reranker"])
    brain.eval()
    brain.retriever.reranker.eval()

    samples = load_dureader(max_n=20000)
    test = samples[-971:]

    print("[prep] 编码所有 passage...")
    all_passages = [s["passage"] for s in samples]
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(all_passages), 32):
            batch = all_passages[i:i+32]
            ids_list = [tok.encode(p, add_bos=True,
                                   max_len=cfg.max_seq_len) for p in batch]
            padded, mask = pad_with_mask(ids_list, cfg.max_seq_len, device)
            h = brain.backbone(padded)["hidden"]
            m = mask.unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1)
            all_embs.append(e.cpu())
    all_embs = torch.cat(all_embs, dim=0)
    print(f"[prep] {len(all_embs)} 个 passage 编码完成")

    top1, top5, mrr_sum, n_valid = 0, 0, 0.0, 0
    cosine_top1 = 0
    for s in test:
        q = s["question"]
        gt = s["passage"]

        q_ids = tok.encode(q, add_bos=True, max_len=cfg.max_seq_len)
        q_x = torch.tensor([q_ids], dtype=torch.long, device=device)
        q_mask = (q_x != 0)
        with torch.no_grad():
            q_h = brain.backbone(q_x)["hidden"]
            q_emb = (q_h * q_mask.unsqueeze(-1).float()).sum(1) \
                    / q_mask.sum(1, keepdim=True).clamp(min=1).float()

        q_n = F.normalize(q_emb.cpu(), dim=-1)
        e_n = F.normalize(all_embs, dim=-1)
        sims = (q_n @ e_n.T).squeeze(0)

        try:
            gt_idx = all_passages.index(gt)
        except ValueError:
            continue

        top50_sims, top50_idx = sims.topk(50)
        top50_idx_list = top50_idx.tolist()
        if gt_idx not in top50_idx_list:
            continue

        n_valid += 1
        if top50_idx_list[0] == gt_idx:
            cosine_top1 += 1

        cand_idx = [gt_idx] + [i for i in top50_idx_list if i != gt_idx]
        cand_passages = [all_passages[i] for i in cand_idx]

        with torch.no_grad():
            cand_ids_list = [tok.encode(c, add_bos=True,
                                        max_len=cfg.max_seq_len)
                             for c in cand_passages]
            padded, c_mask = pad_with_mask(cand_ids_list,
                                           cfg.max_seq_len, device)
            cand_h = brain.backbone(padded)["hidden"]      # [K, T, d]

            scores = brain.retriever.reranker(
                q_h, cand_h, q_mask=q_mask, c_mask=c_mask)  # [1, K]

        _, sorted_idx = scores[0].topk(len(cand_idx))
        rank_t = (sorted_idx == 0).nonzero().flatten()
        rank = int(rank_t[0].item()) if rank_t.numel() > 0 else 999

        if rank == 0:
            top1 += 1
        if rank < 5:
            top5 += 1
        mrr_sum += 1.0 / (rank + 1)

    n = max(n_valid, 1)
    print(f"\n=== Hard Negative 测试 v2 (候选来自 cosine top-50) ===")
    print(f"有效样本: {n_valid}")
    print(f"cosine Top-1 基线: {cosine_top1/n:.3f}")
    print(f"reranker Top-1:    {top1/n:.3f}")
    print(f"reranker Top-5:    {top5/n:.3f}")
    print(f"reranker MRR:      {mrr_sum/n:.3f}")
    print(f"\n随机基线 Top-1 = {1/50:.3f}")


if __name__ == "__main__":
    main()
