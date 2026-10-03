"""微调意图检测 + 行为预测头。解冻 backbone 后 4 层。"""
import json
import os
import sys
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from tokenizer import CharTokenizer
from model.brain import Brain
from model.heads import INTENT_NAMES, BEHAVIOR_NAMES, SENTIMENT_NAMES


# 解冻的层号（15 层模型，解冻后 4 层）
UNFREEZE_LAYERS = []


def setup_trainable(brain, unfreeze_layers=UNFREEZE_LAYERS):
    """冻结前 N-4 层，解冻后 4 层 + heads。"""
    for name, p in brain.backbone.named_parameters():
        # embed 和 norm_f 永远冻结
        if "embed" in name or "norm_f" in name:
            p.requires_grad = False
            continue
        # 后 N 层解冻
        unfreeze = any(f"layers.{i}." in name for i in unfreeze_layers)
        p.requires_grad = unfreeze
    # heads 全部可训
    for p in brain.heads.parameters():
        p.requires_grad = True

    n_backbone = sum(p.numel() for p in brain.backbone.parameters()
                     if p.requires_grad)
    n_heads = sum(p.numel() for p in brain.heads.parameters()
                  if p.requires_grad)
    print(f"[trainable] backbone 后 {len(unfreeze_layers)} 层: "
          f"{n_backbone/1e6:.2f}M, heads: {n_heads/1e6:.2f}M")


class IntentDataset(Dataset):
    def __init__(self, samples, tok, max_len):
        self.samples = samples
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ids = self.tok.encode(s["text"], add_bos=True, max_len=self.max_len)
        return torch.tensor(ids, dtype=torch.long), INTENT_NAMES.index(s["intent"])


class BehaviorDataset(Dataset):
    def __init__(self, samples, tok, max_len):
        self.samples = samples
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x_ids = self.tok.encode(s["input"], add_bos=True, max_len=self.max_len)
        y = BEHAVIOR_NAMES.index(s["label"])
        return torch.tensor(x_ids, dtype=torch.long), y


class SentimentDataset(Dataset):
    def __init__(self, samples, tok, max_len):
        self.samples = samples
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ids = self.tok.encode(s["text"], add_bos=True, max_len=self.max_len)
        return torch.tensor(ids, dtype=torch.long), SENTIMENT_NAMES.index(s["sentiment"])


def collate(batch, pad_id=0):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, :len(x)] = x
    return padded, torch.tensor(ys, dtype=torch.long)


def train_intent(brain, samples, tok, cfg, device, n_epochs=20):
    ds = IntentDataset(samples, tok, cfg.max_seq_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=collate)

    setup_trainable(brain)

    trainable = [p for p in brain.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr)

    best_acc = 0.0
    for epoch in range(n_epochs):
        brain.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for step, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            out = brain(x, need_heads=True)
            logits = out["intent_logits"]
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            acc = (logits.argmax(-1) == y).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        avg_acc = total_acc / max(n, 1)
        best_acc = max(best_acc, avg_acc)
        print(f"[intent e{epoch:>2}] avg_loss={avg_loss:.3f} "
              f"avg_acc={avg_acc:.3f}")

    print(f"[intent] best_acc={best_acc:.3f}")


def train_behavior(brain, samples, tok, cfg, device, n_epochs=20):
    ds = BehaviorDataset(samples, tok, cfg.max_seq_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=collate)

    setup_trainable(brain)

    trainable = [p for p in brain.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr)

    best_acc = 0.0
    for epoch in range(n_epochs):
        brain.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for step, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            out = brain(x, need_heads=True)
            logits = out["behavior_logits"]
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            acc = (logits.argmax(-1) == y).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        avg_acc = total_acc / max(n, 1)
        best_acc = max(best_acc, avg_acc)
        print(f"[behavior e{epoch:>2}] avg_loss={avg_loss:.3f} "
              f"avg_acc={avg_acc:.3f}")

    print(f"[behavior] best_acc={best_acc:.3f}")


def train_sentiment(brain, samples, tok, cfg, device, n_epochs=20):
    ds = SentimentDataset(samples, tok, cfg.max_seq_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=collate)

    setup_trainable(brain)

    trainable = [p for p in brain.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr)

    best_acc = 0.0
    for epoch in range(n_epochs):
        brain.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for step, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            out = brain(x, need_heads=True)
            logits = out["sentiment_logits"]
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            acc = (logits.argmax(-1) == y).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        avg_acc = total_acc / max(n, 1)
        best_acc = max(best_acc, avg_acc)
        print(f"[sentiment e{epoch:>2}] avg_loss={avg_loss:.3f} "
              f"avg_acc={avg_acc:.3f}")

    print(f"[sentiment] best_acc={best_acc:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/mnt/workspace/checkpoints/final.pt")
    parser.add_argument("--intent-data",
                        default="/mnt/workspace/data/intent_dataset_v3.json")
    parser.add_argument("--behavior-data",
                        default="/mnt/workspace/data/behavior_dataset_v2.json")
    parser.add_argument("--sentiment-data",
                        default="/mnt/workspace/data/sentiment_dataset.json")
    parser.add_argument("--save", default="/mnt/workspace/checkpoints/m1_heads_v3.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", default=None,
                        choices=["intent", "behavior", "sentiment"],
                        help="只训练指定的头")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    cfg = Config()
    cfg.lr = args.lr
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  lr={cfg.lr}  epochs={args.epochs}")

    tok = CharTokenizer.from_vocab_json(
        os.path.join(cfg.data_dir, "all_chars_vocab.json"))
    cfg.vocab_size = len(tok)
    print(f"[vocab] {cfg.vocab_size}")

    brain = Brain(cfg).to(device)

    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        backbone_state = {
            k.replace("backbone.", ""): v
            for k, v in ckpt["model"].items()
            if k.startswith("backbone.")
        }
        if not backbone_state:
            backbone_state = ckpt["model"]
        missing, unexpected = brain.backbone.load_state_dict(
            backbone_state, strict=False)
        print(f"[ckpt] {args.ckpt} (missing={len(missing)}, "
              f"unexpected={len(unexpected)})")

    # 加载已有的 heads（如果存在）
    if os.path.exists(args.save):
        try:
            old_ckpt = torch.load(args.save, map_location=device)
            if "heads" in old_ckpt:
                brain.heads.load_state_dict(old_ckpt["heads"], strict=False)
                print(f"[load heads] {args.save}")
        except Exception as e:
            print(f"[load heads] 跳过 ({e})")

    train_intent_flag = args.only in (None, "intent")
    train_behavior_flag = args.only in (None, "behavior")
    train_sentiment_flag = args.only in (None, "sentiment")

    if train_intent_flag and os.path.exists(args.intent_data):
        with open(args.intent_data, encoding="utf-8") as f:
            samples = json.load(f)["samples"]
        print(f"\n[intent] {len(samples)} 条")
        train_intent(brain, samples, tok, cfg, device, args.epochs)

    if train_behavior_flag and os.path.exists(args.behavior_data):
        with open(args.behavior_data, encoding="utf-8") as f:
            samples = json.load(f)["samples"]
        print(f"\n[behavior] {len(samples)} 条")
        train_behavior(brain, samples, tok, cfg, device, args.epochs)

    if train_sentiment_flag and os.path.exists(args.sentiment_data):
        with open(args.sentiment_data, encoding="utf-8") as f:
            samples = json.load(f)["samples"]
        print(f"\n[sentiment] {len(samples)} 条")
        train_sentiment(brain, samples, tok, cfg, device, args.epochs)

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({
        "backbone": brain.backbone.state_dict(),
        "heads": brain.heads.state_dict(),
    }, args.save)
    print(f"\n[save] {args.save}")


if __name__ == "__main__":
    main()
