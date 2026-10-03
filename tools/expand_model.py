import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""深度扩展：Net2DeeperNet。新层输出投影归零 → 恒等映射。"""
import copy
import torch
import torch.nn as nn
from model.backbone import Backbone


def expand_depth(model, n_new_per_gap=1):
    cfg = model.cfg
    n_old = len(model.layers)
    template = model.layers[n_old // 2]

    new_layers, new_heads = [], []
    for i in range(n_old):
        new_layers.append(model.layers[i])
        new_heads.append(model.exit_heads[i])
        if i < n_old - 1:
            for _ in range(n_new_per_gap):
                nl = copy.deepcopy(template)
                with torch.no_grad():
                    nl.attn.o_proj.weight.zero_()
                    nl.mlp.down.weight.zero_()
                new_layers.append(nl)
                nh = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
                if cfg.tie_word_embeddings:
                    nh.weight = model.embed.weight
                new_heads.append(nh)

    model.layers = nn.ModuleList(new_layers)
    model.exit_heads = nn.ModuleList(new_heads)
    model.cfg.n_layers = len(model.layers)
    print(f"[expand] 层数 {n_old} → {len(model.layers)}")
    return model


if __name__ == "__main__":
    import argparse
    from config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-per-gap", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = Config()

    # 从词表读实际 vocab_size
    from tokenizer import CharTokenizer
    vocab_path = os.path.join(cfg.data_dir, "all_chars_vocab.json")
    tok = CharTokenizer.from_vocab_json(vocab_path)
    cfg.vocab_size = len(tok)
    print(f"[vocab] {cfg.vocab_size}")

    model = Backbone(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[load] step={ckpt.get('step', '?')}")

    model = expand_depth(model, n_new_per_gap=args.n_per_gap)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] 参数量 {n_params/1e6:.2f}M")

    # 验证恒等
    model.eval()
    x = torch.randint(1, 1000, (2, 32)).to(device)
    with torch.no_grad():
        out = model(x)
        print(f"[verify] logits finite: {torch.isfinite(out['final_logits']).all().item()}")

    torch.save({
        "model": model.state_dict(),
        "config": model.cfg.__dict__,
        "step": ckpt.get("step", 0),
    }, args.out)
    print(f"[save] {args.out}")
