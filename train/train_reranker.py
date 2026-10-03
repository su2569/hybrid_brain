"""训练 reranker：pairwise ranking loss。"""
import os, sys, pickle, random, time, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from tokenizer import CharTokenizer
from model.brain import Brain
from memory.knowledge import KnowledgeBase
from data.dataset import load_corpus


class PairwiseDataset(Dataset):
    """(query, positive, negative) 三元组"""
    def __init__(self, triples, tok, max_len):
        self.triples = triples
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, i):
        t = self.triples[i]
        def enc(x):
            return self.tok.encode(x, add_bos=True, max_len=self.max_len)
        return (enc(t["query"]), enc(t["pos"]), enc(t["neg"]))


def collate(batch, pad_id=0):
    """每个样本的 q/pos/neg 独立 pad。"""
    def pad_list(seqs):
        max_len = max(len(s) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s)
        return out

    qs, ps, ns = zip(*batch)
    return pad_list(qs), pad_list(ps), pad_list(ns)


def build_triples(texts, max_triples=10000):
    triples = []
    for text in texts:
        lines = [l for l in text.split("\n") if l.strip()]
        for i in range(len(lines) - 2):
            q = lines[i]
            pos = lines[i + 1]
            # 从其他行随机挑一个负样本
            neg_candidates = [l for l in lines if l != pos and l != q]
            if not neg_candidates:
                continue
            neg = random.choice(neg_candidates)
            triples.append({"query": q, "pos": pos, "neg": neg})
            if len(triples) >= max_triples:
                return triples
    return triples


@torch.no_grad()
def get_token_hidden(brain, tok, text, device, max_len):
    ids = tok.encode(text, add_bos=True, max_len=max_len)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = brain.backbone(x)
    return out["hidden"][0].cpu()   # [T, d]


def train_reranker(brain, triples, tok, cfg, device, epochs=10, lr=1e-4):
    # 冻结其他，只训 reranker
    for p in brain.parameters():
        p.requires_grad = False
    for p in brain.retriever.reranker.parameters():
        p.requires_grad = True

    n_params = sum(p.numel() for p in brain.retriever.reranker.parameters())
    print(f"[trainable] reranker: {n_params/1e6:.2f}M")

    ds = PairwiseDataset(triples, tok, cfg.max_seq_len)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate,
                    num_workers=2)

    opt = torch.optim.AdamW(brain.retriever.reranker.parameters(), lr=lr)
    brain.backbone.eval()

    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        for q_ids, p_ids, n_ids in dl:
            q_ids = q_ids.to(device)
            p_ids = p_ids.to(device)
            n_ids = n_ids.to(device)

            # 前向拿 token hidden
            with torch.no_grad():
                q_h = brain.backbone(q_ids)["hidden"]    # [B, T_q, d]
                p_h = brain.backbone(p_ids)["hidden"]    # [B, T_p, d]
                n_h = brain.backbone(n_ids)["hidden"]    # [B, T_n, d]

            # Reranker 打分：query 和 pos/neg 的相似度
            # 把 pos/neg 当作候选，K=B
            # 简化：逐个打分（也可以用 batch 化）
            pos_scores = []
            neg_scores = []
            for i in range(q_h.size(0)):
                # query: [1, T_q, d]
                qi = q_h[i:i+1]
                # pos: [1, T_p, d]
                pi = p_h[i:i+1]
                ni = n_h[i:i+1]
                ps = brain.retriever.reranker(qi, pi)   # [1, 1]
                ns = brain.retriever.reranker(qi, ni)   # [1, 1]
                pos_scores.append(ps.squeeze())
                neg_scores.append(ns.squeeze())

            pos_scores = torch.stack(pos_scores)
            neg_scores = torch.stack(neg_scores)

            loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n += 1
            step += 1

            if step % 50 == 0:
                dt = time.time() - t0
                print(f"[step {step:>5}] loss={total_loss/n:.4f} elapsed={dt:.0f}s")
                total_loss, n = 0.0, 0

        print(f"[epoch {epoch}] done, step={step}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/mnt/workspace/checkpoints/final.pt")
    parser.add_argument("--retriever", default="/mnt/workspace/checkpoints/m2_retriever_v4.pt")
    parser.add_argument("--kb", default="/mnt/workspace/checkpoints/m2_kb_v4.pkl")
    parser.add_argument("--save", default="/mnt/workspace/checkpoints/m2_reranker.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device("cuda")
    tok = CharTokenizer.from_vocab_json("data/all_chars_vocab.json")
    cfg.vocab_size = len(tok)

    brain = Brain(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    brain.backbone.load_state_dict(ckpt["model"], strict=False)

    if os.path.exists(args.retriever):
        rckpt = torch.load(args.retriever, map_location=device)
        brain.retriever.load_state_dict(rckpt["retriever"], strict=False)

    with open(args.kb, "rb") as f:
        kb_data = pickle.load(f)
    kb = KnowledgeBase(dim=cfg.d_model)
    kb.nodes = kb_data["nodes"]
    brain.attach_kb(kb)

    # 构造 pairwise 数据
    print("[data] 加载 LCCC...")
    texts = load_corpus("data/lccc_50k.json", max_samples=5000,
                        assistant_name="昔涟")
    triples = build_triples(texts, max_triples=10000)
    print(f"[triples] {len(triples)} 个")

    train_reranker(brain, triples, tok, cfg, device,
                   epochs=args.epochs, lr=args.lr)

    torch.save({
        "reranker": brain.retriever.reranker.state_dict(),
    }, args.save)
    print(f"[save] {args.save}")


if __name__ == "__main__":
    main()
