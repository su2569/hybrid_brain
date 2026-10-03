"""用 Qwen 骨干重训三头。只训头，Qwen 全冻结。"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config_qwen import ConfigQwen
from tokenizer_qwen import QwenTokenizer
from model.backbone_qwen import QwenBackbone
from model.heads import IntentAndPredictHead, INTENT_NAMES, \
    SENTIMENT_NAMES, BEHAVIOR_NAMES


class HeadDataset(Dataset):
    def __init__(self, samples, tok, max_len, task):
        self.samples = samples
        self.tok = tok
        self.max_len = max_len
        self.task = task

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        text = s.get("text") or s.get("input") or ""
        ids = self.tok.encode(text, max_len=self.max_len)
        if self.task == "intent":
            y = INTENT_NAMES.index(s["intent"])
        elif self.task == "sentiment":
            y = SENTIMENT_NAMES.index(s["sentiment"])
        elif self.task == "behavior":
            y = BEHAVIOR_NAMES.index(s["label"])
        return torch.tensor(ids, dtype=torch.long), y


def collate(batch, pad_id):
    xs, ys = zip(*batch)
    L = max(len(x) for x in xs)
    padded = torch.full((len(xs), L), pad_id, dtype=torch.long)
    mask = torch.zeros(len(xs), L, dtype=torch.bool)
    for i, x in enumerate(xs):
        padded[i, :len(x)] = x
        mask[i, :len(x)] = True
    return padded, mask, torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def extract_features(backbone, tok, samples, cfg, device, task):
    """一次性提取所有样本的 pooled hidden，避免反复前向。"""
    ds = HeadDataset(samples, tok, cfg.max_seq_len, task)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                    collate_fn=lambda b: collate(b, tok.PAD))
    feats, labels = [], []
    for x, mask, y in dl:
        x = x.to(device); mask = mask.to(device)
        h = backbone(x)["hidden"]                    # [B, T, 896]
        m = mask.unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1)
        feats.append(e.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def train_head(feats, labels, n_classes, device, epochs=30, lr=5e-4,
               batch=64, dropout=0.3):
    d = feats.size(-1)
    head = torch.nn.Sequential(
        torch.nn.LayerNorm(d),
        torch.nn.Linear(d, d),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(d, d),
        torch.nn.Linear(d, n_classes),
    ).to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=lr,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    n = len(feats)
    idx = torch.randperm(n)
    n_val = max(1, n // 10)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_tr = feats[tr_idx].to(device)
    y_tr = labels[tr_idx].to(device)
    X_val = feats[val_idx].to(device)
    y_val = labels[val_idx].to(device)

    best_acc = 0.0
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(X_tr))
        tot_loss, tot_acc, cnt = 0.0, 0.0, 0
        for i in range(0, len(X_tr), batch):
            bi = perm[i:i+batch]
            logits = head(X_tr[bi])
            loss = F.cross_entropy(logits, y_tr[bi])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item()
            tot_acc += (logits.argmax(-1) == y_tr[bi]).float().mean().item()
            cnt += 1
        sched.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(X_val)
            val_acc = (val_logits.argmax(-1) == y_val).float().mean().item()

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[ep {ep+1:>2}] tr_loss={tot_loss/cnt:.3f} "
                  f"tr_acc={tot_acc/cnt:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    print(f"[best] val_acc={best_acc:.3f}")
    head.load_state_dict(best_state)
    return head, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent-data",
                    default="/mnt/workspace/data/intent_dataset_v4.json")
    ap.add_argument("--sentiment-data",
                    default="/mnt/workspace/data/sentiment_dataset.json")
    ap.add_argument("--behavior-data",
                    default="/mnt/workspace/data/behavior_dataset_v2.json")
    ap.add_argument("--save", default="/mnt/workspace/checkpoints/heads_qwen.pt")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    cfg = ConfigQwen()
    device = torch.device("cuda")

    print("[load] QwenBackbone...")
    backbone = QwenBackbone(cfg.qwen_path, freeze=True,
                            dtype=torch.bfloat16).to(device)
    backbone.eval()

    print("[load] QwenTokenizer...")
    tok = QwenTokenizer(cfg.qwen_path)

    results = {}

    # ---------- 意图 ----------
    if os.path.exists(args.intent_data):
        print(f"\n=== 意图 ===")
        with open(args.intent_data) as f:
            samples = json.load(f)["samples"]
        print(f"[data] {len(samples)} 条")
        feats, labels = extract_features(backbone, tok, samples, cfg,
                                         device, "intent")
        head, acc = train_head(feats, labels, len(INTENT_NAMES), device,
                               epochs=args.epochs, lr=cfg.lr_heads)
        results["intent"] = {"state": head.state_dict(), "acc": acc}

    # ---------- 情感 ----------
    if os.path.exists(args.sentiment_data):
        print(f"\n=== 情感 ===")
        with open(args.sentiment_data) as f:
            samples = json.load(f)["samples"]
        print(f"[data] {len(samples)} 条")
        feats, labels = extract_features(backbone, tok, samples, cfg,
                                         device, "sentiment")
        head, acc = train_head(feats, labels, len(SENTIMENT_NAMES), device,
                               epochs=args.epochs, lr=cfg.lr_heads)
        results["sentiment"] = {"state": head.state_dict(), "acc": acc}

    # ---------- 行为 ----------
    if os.path.exists(args.behavior_data):
        print(f"\n=== 行为 ===")
        with open(args.behavior_data) as f:
            samples = json.load(f)["samples"]
        print(f"[data] {len(samples)} 条")
        feats, labels = extract_features(backbone, tok, samples, cfg,
                                         device, "behavior")
        head, acc = train_head(feats, labels, len(BEHAVIOR_NAMES), device,
                               epochs=args.epochs, lr=cfg.lr_heads)
        results["behavior"] = {"state": head.state_dict(), "acc": acc}

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({
        "heads": {k: v["state"] for k, v in results.items()},
        "acc": {k: v["acc"] for k, v in results.items()},
        "d_model": cfg.d_model,
    }, args.save)
    print(f"\n[save] {args.save}")
    for k, v in results.items():
        print(f"  {k}: acc={v['acc']:.3f}")


if __name__ == "__main__":
    main()
