"""Qwen 骨干 + 新三头 + 新 reranker 的端到端验证。"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from config_qwen import ConfigQwen
from tokenizer_qwen import QwenTokenizer
from model.backbone_qwen import QwenBackbone
from memory.reranker import Reranker
from model.heads import INTENT_NAMES
from rag.kb_builder import load_dureader


def pad_batch(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=device)
    m = torch.zeros(len(seqs), L, dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, device=device)
        m[i, :len(s)] = True
    return out, m


@torch.no_grad()
def encode_pooled(backbone, tok, texts, cfg, device, batch=16):
    embs = []
    for i in range(0, len(texts), batch):
        b = texts[i:i+batch]
        ids = [tok.encode(t, max_len=cfg.max_seq_len) for t in b]
        x, mask = pad_batch(ids, tok.PAD, device)
        h = backbone(x)["hidden"]
        m = mask.unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1)
        embs.append(e.cpu())
    return torch.cat(embs, dim=0)


def main():
    cfg = ConfigQwen()
    device = torch.device("cuda")

    print("[load] Qwen...")
    backbone = QwenBackbone(cfg.qwen_path, freeze=True).to(device).eval()
    tok = QwenTokenizer(cfg.qwen_path)

    print("[load] reranker...")
    reranker_path = "/mnt/workspace/checkpoints/reranker_qwen.pt"
    if os.path.exists(reranker_path):
        reranker = Reranker(cfg.d_model, cfg.reranker_heads,
                            hidden=cfg.reranker_hidden).to(device)
        ck = torch.load(reranker_path, map_location=device)
        reranker.load_state_dict(ck["reranker"])
        reranker.eval()
        use_reranker = True
        print(f"[reranker] 加载 {reranker_path}")
    else:
        reranker = None
        use_reranker = False
        print(f"[reranker] 不存在 {reranker_path}，跳过 reranker 评测")

    print("[load] heads...")
    hk = torch.load("/mnt/workspace/checkpoints/heads_qwen.pt",
                    map_location=device)
    print("  heads acc:", hk["acc"])

    samples = load_dureader(max_n=20000)
    test = samples[-971:]
    all_passages = [s["passage"] for s in samples]

    print("[prep] 编码全库...")
    all_embs = encode_pooled(backbone, tok, all_passages, cfg, device)
    e_n = F.normalize(all_embs, dim=-1)

    top1_cos = top1_rr = top5_rr = 0
    mrr_cos = mrr_rr = 0.0
    n_valid = 0

    for s in test:
        gt = s["passage"]
        try:
            gt_idx = all_passages.index(gt)
        except ValueError:
            continue

        q_ids = tok.encode(s["question"], max_len=cfg.max_seq_len)
        qx, qm = pad_batch([q_ids], tok.PAD, device)
        with torch.no_grad():
            q_h = backbone(qx)["hidden"].float()
            m = qm.unsqueeze(-1).float()
            q_emb = (q_h * m).sum(1) / m.sum(1).clamp(min=1)

        sims = (F.normalize(q_emb.cpu(), dim=-1) @ e_n.T).squeeze(0)
        top20_sims, top20_idx = sims.topk(20)
        top20_idx_list = top20_idx.tolist()
        if gt_idx not in top20_idx_list:
            continue
        n_valid += 1

        # cosine 排名
        cos_rank = top20_idx_list.index(gt_idx)
        if cos_rank == 0: top1_cos += 1
        mrr_cos += 1.0 / (cos_rank + 1)

        # reranker 排名（可选）
        if not use_reranker:
            continue
        cand_idx = [gt_idx] + [i for i in top20_idx_list if i != gt_idx]
        cand_texts = [all_passages[i] for i in cand_idx]
        c_ids = [tok.encode(c, max_len=cfg.max_seq_len) for c in cand_texts]
        c_pad, c_mask = pad_batch(c_ids, tok.PAD, device)

        with torch.no_grad():
            c_h = backbone(c_pad)["hidden"].float()
            scores = reranker(q_h, c_h, q_mask=qm, c_mask=c_mask)[0]

        _, sorted_idx = scores.topk(len(cand_idx))
        rr_rank = (sorted_idx == 0).nonzero().flatten()[0].item()
        if rr_rank == 0: top1_rr += 1
        if rr_rank < 5: top5_rr += 1
        mrr_rr += 1.0 / (rr_rank + 1)

    n = max(n_valid, 1)
    print(f"\n=== 端到端验证 (Qwen2.5-0.5B) ===")
    print(f"有效样本: {n_valid}/{len(test)}")
    print(f"\n--- 纯 cosine ---")
    print(f"Top-1: {top1_cos/n:.3f}")
    print(f"MRR:   {mrr_cos/n:.3f}")
    if use_reranker:
        print(f"\n--- Qwen cosine + 新 reranker ---")
        print(f"Top-1: {top1_rr/n:.3f}")
        print(f"Top-5: {top5_rr/n:.3f}")
        print(f"MRR:   {mrr_rr/n:.3f}")
    else:
        print(f"\n--- reranker 未加载，跳过 ---")
    print(f"\n--- 对照（旧 73M 字符级）---")
    print(f"旧 cosine Top-1: 0.190")
    print(f"旧 reranker Top-1: 0.107")


if __name__ == "__main__":
    main()
