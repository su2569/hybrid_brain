"""只重训 intent 头。加载已有 heads，合并保存。"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config_qwen import ConfigQwen
from tokenizer_qwen import QwenTokenizer
from model.backbone_qwen import QwenBackbone
from model.heads import INTENT_NAMES
from train.train_heads_qwen import extract_features, train_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default="/mnt/workspace/data/intent_dataset_v5.json")
    ap.add_argument("--save",
                    default="/mnt/workspace/checkpoints/heads_qwen.pt")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    cfg = ConfigQwen()
    device = torch.device("cuda")

    print("[load] QwenBackbone...")
    backbone = QwenBackbone(cfg.qwen_path, freeze=True,
                            dtype=torch.bfloat16).to(device)
    backbone.eval()
    tok = QwenTokenizer(cfg.qwen_path)

    with open(args.data, encoding="utf-8") as f:
        samples = json.load(f)["samples"]
    print(f"[data] {len(samples)} 条")
    from collections import Counter
    c = Counter(s["intent"] for s in samples)
    for k in INTENT_NAMES:
        print(f"  {k}: {c.get(k, 0)}")

    print("\n[extract] 编码所有样本...")
    feats, labels = extract_features(backbone, tok, samples, cfg,
                                     device, "intent")

    print("\n[train] intent head")
    head, acc = train_head(feats, labels, len(INTENT_NAMES), device,
                           epochs=args.epochs, lr=cfg.lr_heads)
    print(f"[best] val_acc={acc:.3f}")

    print(f"\n[load] 已有 heads: {args.save}")
    old = torch.load(args.save, map_location=device)
    old_heads = old["heads"]
    old_acc = old.get("acc", {})

    old_heads["intent"] = {k: v.cpu() for k, v in head.state_dict().items()}
    old_acc["intent"] = acc

    torch.save({
        "heads": old_heads,
        "acc": old_acc,
        "d_model": cfg.d_model,
    }, args.save)
    print(f"[save] {args.save}")
    print(f"  intent:    {acc:.3f}")
    print(f"  sentiment: {old_acc.get('sentiment', '?')}")
    print(f"  behavior:  {old_acc.get('behavior', '?')}")


if __name__ == "__main__":
    main()
