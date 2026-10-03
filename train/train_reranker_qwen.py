"""用 Qwen 骨干重训 reranker。hard negative + mask。"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config_qwen import ConfigQwen
from tokenizer_qwen import QwenTokenizer
from model.backbone_qwen import QwenBackbone
from memory.reranker import Reranker
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


class QADataset(Dataset):
    def __init__(self, samples, hard_negs, tok, max_len):
        self.samples = samples
        self.hard_negs = hard_negs
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        q = self.tok.encode(s["question"], max_len=self.max_len)
        p = self.tok.encode(s["passage"], max_len=self.max_len)
        negs = [self.tok.encode(
                    self.samples[j]["passage"], max_len=self.max_len)
                for j in self.hard_negs[i]]
        return q, p, negs


def collate(batch, pad_id):
    qs = [b[0] for b in batch]
    ps = [b[1] for b in batch]
    negs_flat = [n for b in batch for n in b[2]]
    K = len(batch[0][2])
    B = len(batch)
    q_pad, q_m = pad_batch(qs, pad_id, "cpu")
    p_pad, p_m = pad_batch(ps, pad_id, "cpu")
    n_pad, n_m = pad_batch(negs_flat, pad_id, "cpu")
    return q_pad, q_m, p_pad, p_m, n_pad, n_m, B, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save",
                    default="/mnt/workspace/checkpoints/reranker_qwen.pt")
    ap.add_argument("--topk-neg", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=5000)
    args = ap.parse_args()

    cfg = ConfigQwen()
    device = torch.device("cuda")

    print("[load] QwenBackbone...")
    backbone = QwenBackbone(cfg.qwen_path, freeze=True).to(device)
    backbone.eval()
    tok = QwenTokenizer(cfg.qwen_path)

    samples = load_dureader(max_n=args.max_samples)
    print(f"[DuReader] {len(samples)} 条")

    # ---- 用 Qwen 挖 hard negative ----
    cache_path = f"/mnt/workspace/checkpoints/emb_cache_{args.max_samples}.pt"
    if os.path.exists(cache_path):
        print(f"[prep] 加载缓存 {cache_path}")
        _c = torch.load(cache_path, map_location="cpu")
        passage_embs = _c["passage_embs"]
        question_embs = _c["question_embs"]
    else:
        print("[prep] 编码全库（Qwen）...")
        all_passages = [s["passage"] for s in samples]
        all_questions = [s["question"] for s in samples]
        passage_embs = encode_pooled(backbone, tok, all_passages, cfg, device)
        question_embs = encode_pooled(backbone, tok, all_questions, cfg, device)
        torch.save({"passage_embs": passage_embs,
                    "question_embs": question_embs}, cache_path)
        print(f"[prep] 缓存到 {cache_path}")
    e_n = F.normalize(passage_embs, dim=-1)
    q_n = F.normalize(question_embs, dim=-1)

    print(f"[prep] 挖 top-100 候选池（用 question 查 passage）...")
    cand_pool = []
    for i in range(len(samples)):
        sims = (q_n[i:i+1] @ e_n.T).squeeze(0)
        sims[i] = -2.0
        _, idx = sims.topk(100)
        cand_pool.append(idx.tolist())
    print(f"[prep] 完成")

    def build_epoch_negs(pool, K, n_fixed, rng):
        """每 epoch: n_fixed 个固定 hard + (K-n_fixed) 个从 top-100 随机抽"""
        out = []
        for p in pool:
            fixed = p[:n_fixed]
            rest = p[n_fixed:]
            need = K - n_fixed
            if len(rest) >= need:
                r = rng.sample(rest, need)
            else:
                r = rest
            out.append(fixed + r)
        return out

    # ---- Reranker 模型 ----
    reranker = Reranker(cfg.d_model, cfg.reranker_heads,
                        hidden=cfg.reranker_hidden).to(device)
    # 热启动：加载上一轮训练结果
    prev_ckpt = "/mnt/workspace/checkpoints/reranker_qwen_v1.pt"
    if os.path.exists(prev_ckpt):
        _ck = torch.load(prev_ckpt, map_location=device)
        if "reranker" in _ck:
            reranker.load_state_dict(_ck["reranker"])
            print(f"[warm-start] loaded from {prev_ckpt}")
        else:
            print(f"[warn] ckpt 里没 reranker key")
    else:
        print(f"[warn] 未找到 {prev_ckpt}，从零开始")

    n = sum(p.numel() for p in reranker.parameters())
    print(f"[model] reranker {n/1e6:.2f}M 参数")

    opt = torch.optim.AdamW(reranker.parameters(), lr=args.lr)

    import random as _pyrandom
    rng = _pyrandom.Random(42)

    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        ep_negs = build_epoch_negs(cand_pool, args.topk_neg,
                                   n_fixed=10, rng=rng)
        ds = QADataset(samples, ep_negs, tok, cfg.max_seq_len)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.PAD))
        n_random = args.topk_neg - 10
        print(f"[epoch {ep}] negs: 10 fixed + {n_random} random")
        tot_loss, tot_acc, cnt = 0.0, 0.0, 0
        for q_pad, q_m, p_pad, p_m, n_pad, n_m, B, K in dl:
            q_pad, q_m = q_pad.to(device), q_m.to(device)
            p_pad, p_m = p_pad.to(device), p_m.to(device)
            n_pad, n_m = n_pad.to(device), n_m.to(device)

            with torch.no_grad():
                q_h = backbone(q_pad)["hidden"].float()        # [B, T_q, d]
                p_h = backbone(p_pad)["hidden"].float()        # [B, T_p, d]
                n_h = backbone(n_pad)["hidden"].float()        # [B*K, T_n, d]

            pos_s, neg_s = [], []
            for b in range(B):
                qi = q_h[b:b+1]; qmi = q_m[b:b+1]
                pi = p_h[b:b+1]; pmi = p_m[b:b+1]
                ni = n_h[b*K:(b+1)*K]; nmi = n_m[b*K:(b+1)*K]
                # pos 和 negs 先 pad 到相同 T，再拼接
                Tp = pi.size(1)
                Tn = ni.size(1)
                T = max(Tp, Tn)
                if Tp < T:
                    pad_h = torch.zeros(1, T - Tp, pi.size(2),
                                        dtype=pi.dtype, device=pi.device)
                    pad_m = torch.zeros(1, T - Tp,
                                        dtype=pmi.dtype, device=pmi.device)
                    pi = torch.cat([pi, pad_h], dim=1)
                    pmi = torch.cat([pmi, pad_m], dim=1)
                if Tn < T:
                    pad_h = torch.zeros(K, T - Tn, ni.size(2),
                                        dtype=ni.dtype, device=ni.device)
                    pad_m = torch.zeros(K, T - Tn,
                                        dtype=nmi.dtype, device=nmi.device)
                    ni = torch.cat([ni, pad_h], dim=1)
                    nmi = torch.cat([nmi, pad_m], dim=1)
                cand_h = torch.cat([pi, ni], dim=0)      # [1+K, T, d]
                cand_m = torch.cat([pmi, nmi], dim=0)    # [1+K, T]
                scores = reranker(qi, cand_h, q_mask=qmi, c_mask=cand_m)
                pos_s.append(scores[0, 0])
                neg_s.append(scores[0, 1:])

            pos_s = torch.stack(pos_s)                 # [B]
            neg_s = torch.stack(neg_s)                 # [B, K]

            scores = torch.cat([pos_s.unsqueeze(-1), neg_s], dim=-1)  # [B, 1+K]
            labels = torch.zeros(scores.size(0), dtype=torch.long,
                                 device=scores.device)
            loss_list = F.cross_entropy(scores, labels)
            loss_pair = -F.logsigmoid(pos_s.unsqueeze(-1) - neg_s).mean()
            loss = 0.3 * loss_list + loss_pair
            acc = (scores.argmax(-1) == 0).float().mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
            opt.step()

            step += 1
            tot_loss += loss.item()
            tot_acc += acc.item()
            cnt += 1

            if step % 50 == 0:
                dt = time.time() - t0
                print(f"[step {step:>5}] loss={tot_loss/cnt:.4f} "
                      f"acc={tot_acc/cnt:.3f} elapsed={dt:.0f}s")
                tot_loss, tot_acc, cnt = 0.0, 0.0, 0

        print(f"[epoch {ep}] done, step={step}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({
        "reranker": reranker.state_dict(),
        "d_model": cfg.d_model,
    }, args.save)
    print(f"[save] {args.save}")


if __name__ == "__main__":
    main()
