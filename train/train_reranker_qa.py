"""用 DuReader 训练 reranker：pairwise ranking。"""
import os, sys, csv, random, time, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from tokenizer import CharTokenizer
from model.brain import Brain
from rag.kb_builder import load_dureader


class QADataset(Dataset):
    def __init__(self, samples, tok, max_len):
        self.samples = samples
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        q = self.tok.encode(s["question"], add_bos=True, max_len=self.max_len)
        p = self.tok.encode(s["passage"], add_bos=True, max_len=self.max_len)
        j = random.randint(0, len(self.samples) - 1)
        n = self.tok.encode(self.samples[j]["passage"],
                            add_bos=True, max_len=self.max_len)
        return q, p, n


def collate(batch, pad_id=0):
    def pad(seqs):
        L = max(len(s) for s in seqs)
        out = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s)
        return out
    qs, ps, ns = zip(*batch)
    return pad(qs), pad(ps), pad(ns)


def train(brain, samples, tok, cfg, device, epochs=5, lr=1e-4):
    for p in brain.parameters():
        p.requires_grad = False
    for p in brain.retriever.reranker.parameters():
        p.requires_grad = True

    n = sum(p.numel() for p in brain.retriever.reranker.parameters())
    print(f"[trainable] reranker: {n/1e6:.2f}M")

    ds = QADataset(samples, tok, cfg.max_seq_len)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW(brain.retriever.reranker.parameters(), lr=lr)
    brain.backbone.eval()
    brain.retriever.reranker.train()

    step, t0 = 0, time.time()
    for epoch in range(epochs):
        total_loss, total_acc, cnt = 0.0, 0.0, 0
        for q_ids, p_ids, n_ids in dl:
            q_ids = q_ids.to(device)
            p_ids = p_ids.to(device)
            n_ids = n_ids.to(device)
            with torch.no_grad():
                q_h = brain.backbone(q_ids)["hidden"]
                p_h = brain.backbone(p_ids)["hidden"]
                n_h = brain.backbone(n_ids)["hidden"]

            pos_scores, neg_scores = [], []
            for i in range(q_h.size(0)):
                qi = q_h[i:i+1]
                L = min(p_h.size(1), n_h.size(1))
                pi = p_h[i:i+1, :L]
                ni = n_h[i:i+1, :L]
                ps = brain.retriever.reranker(qi, pi)
                ns = brain.retriever.reranker(qi, ni)
                pos_scores.append(ps.reshape(()))
                neg_scores.append(ns.reshape(()))

            pos_scores = torch.stack(pos_scores)
            neg_scores = torch.stack(neg_scores)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            acc = (pos_scores > neg_scores).float().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            total_loss += loss.item()
            total_acc += acc.item()
            cnt += 1

            if step % 100 == 0:
                dt = time.time() - t0
                print(f"[step {step:>5}] loss={total_loss/cnt:.4f} "
                      f"acc={total_acc/cnt:.3f} elapsed={dt:.0f}s")
                total_loss, total_acc, cnt = 0.0, 0.0, 0

        print(f"[epoch {epoch}] done, step={step}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/mnt/workspace/checkpoints/final.pt")
    parser.add_argument("--save", default="/mnt/workspace/checkpoints/qa_reranker.pt")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device("cuda")
    tok = CharTokenizer.from_vocab_json("data/all_chars_vocab.json")
    cfg.vocab_size = len(tok)

    brain = Brain(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = {k.replace("backbone.", ""): v for k, v in ckpt["model"].items()
             if k.startswith("backbone.")}
    brain.backbone.load_state_dict(state if state else ckpt["model"], strict=False)

    samples = load_dureader(max_n=10000)
    print(f"[DuReader] {len(samples)} 条")

    train(brain, samples, tok, cfg, device, epochs=args.epochs)
    torch.save({"reranker": brain.retriever.reranker.state_dict()}, args.save)
    print(f"[save] {args.save}")


if __name__ == "__main__":
    main()
