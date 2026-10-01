"""M0 预训练脚本。

用法：
    cd hybrid_brain
    python train/pretrain.py

特性：
  - 自动检测 GPU / CPU
  - 自动识别多种语料格式（LCCC 原生 / cyrene_clean / JSONL / txt / gz）
  - 昔涟人格数据按比例上采样
  - bf16 混合精度
  - warmup + cosine 调度
  - 每 N 步评估 / 保存
  - 训练中途采样生成，直观观察进度
  - 支持从 checkpoint 恢复
"""
import os
import sys
import math
import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 让脚本能直接跑（把项目根加入 sys.path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from tokenizer import CharTokenizer
from model.backbone import Backbone
from data.dataset import TextDataset, load_corpus


# ============================================================
# 工具
# ============================================================
def get_device(cfg: Config) -> torch.device:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，回退到 CPU")
        return torch.device("cpu")
    return torch.device(cfg.device)


def build_lr_scheduler(opt, cfg: Config, total_steps: int):
    """warmup + cosine"""
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ============================================================
# 数据
# ============================================================
def prepare_data(cfg: Config):
    """加载所有语料，返回 (train_texts, val_texts, tokenizer)"""
    # 1. 分词器
    vocab_path = os.path.join(cfg.data_dir, "all_chars_vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"找不到词表: {vocab_path}")
    tok = CharTokenizer.from_vocab_json(vocab_path, vocab_limit=cfg.vocab_size)
    print(f"[vocab] 加载 {len(tok)} 个字符")

    # 2. LCCC
    print(f"[data] 加载 LCCC: {cfg.lccc_path}")
    lccc = load_corpus(cfg.lccc_path, max_samples=cfg.lccc_max_samples,
                       assistant_name="昔涟")
    print(f"[data] LCCC {len(lccc):,} 条")

    # 3. 昔涟人格数据
    cyrene = []
    if os.path.exists(cfg.cyrene_path):
        cyrene = load_corpus(cfg.cyrene_path, assistant_name="昔涟")
        print(f"[data] 昔涟 {len(cyrene):,} 条")
    else:
        print(f"[warn] 未找到昔涟数据: {cfg.cyrene_path}")

    # 4. 上采样昔涟到目标比例
    if cyrene and lccc:
        n_target = int(len(lccc) * cfg.cyrene_weight / (1 - cfg.cyrene_weight))
        if len(cyrene) < n_target:
            reps = (n_target // len(cyrene)) + 1
            cyrene = (cyrene * reps)[:n_target]
            print(f"[data] 昔涟上采样 → {len(cyrene):,} 条 "
                  f"(权重 ~{cfg.cyrene_weight:.0%})")

    texts = lccc + cyrene
    if not texts:
        raise RuntimeError("没有任何语料，请检查 data_dir 下的数据文件")

    # 5. UNK 率监控（抽前 1000 条）
    sample = texts[:1000]
    unk = sum(tok.unk_rate(t) for t in sample) / max(len(sample), 1)
    print(f"[vocab] UNK rate (前 1000 条): {unk:.4f}")

    # 6. 划分 train / val
    n_val = max(1, int(len(texts) * cfg.val_ratio))
    val_texts = texts[:n_val]
    train_texts = texts[n_val:]
    print(f"[data] train={len(train_texts):,} val={len(val_texts):,}")

    return train_texts, val_texts, tok


# ============================================================
# 评估
# ============================================================
@torch.no_grad()
def evaluate(model: Backbone, val_dl: DataLoader,
             device: torch.device, use_amp: bool) -> float:
    model.eval()
    total_loss = 0.0
    n_batch = 0
    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            out = model(x)
            logits = out["final_logits"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1), ignore_index=-100)
        total_loss += loss.item()
        n_batch += 1
    avg = total_loss / max(n_batch, 1)
    model.train()
    return math.exp(avg)


# ============================================================
# 采样（直观观察训练进度）
# ============================================================
@torch.no_grad()
def sample(model: Backbone, tok: CharTokenizer, prompt: str,
           device: torch.device, max_new: int = 30,
           temperature: float = 0.8, top_k: int = 20) -> str:
    model.eval()
    ids = tok.encode(prompt, add_bos=True)
    ids = ids[-model.cfg.max_seq_len:]
    for _ in range(max_new):
        x = torch.tensor([ids[-model.cfg.max_seq_len:]], dtype=torch.long,
                         device=device)
        out = model(x)
        logits = out["final_logits"][0, -1] / temperature
        # 屏蔽无效 id
        logits[len(tok):] = -float("inf")
        if top_k > 0:
            k = min(top_k, len(tok))
            v, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < v[-1], -float("inf"))
        probs = F.softmax(logits, dim=-1)
        nid = int(torch.multinomial(probs, 1).item())
        ids.append(nid)
        if nid == tok.EOS:
            break
    model.train()
    return tok.decode(ids)


# ============================================================
# 训练
# ============================================================
def train(cfg: Config, resume_from: str = None):
    torch.manual_seed(cfg.seed)
    device = get_device(cfg)
    print(f"[device] {device}")

    # ---------- 数据 ----------
    train_texts, val_texts, tok = prepare_data(cfg)
    train_ds = TextDataset(train_texts, tok, cfg.max_seq_len)
    val_ds = TextDataset(val_texts, tok, cfg.max_seq_len)
    print(f"[data] train samples={len(train_ds):,} "
          f"val samples={len(val_ds):,}")

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                          shuffle=True, num_workers=0,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size,
                        shuffle=False, num_workers=0)

    # ---------- 模型 ----------
    model = Backbone(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e6:.2f}M 参数 "
          f"({cfg.n_layers}×{cfg.d_model})")

    # weight tying：最终 exit_head 与 embed 共享
    if cfg.tie_word_embeddings:
        model.exit_heads[-1].weight = model.embed.weight

    # ---------- 优化器 / 调度器 ----------
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    sched = build_lr_scheduler(opt, cfg, cfg.max_steps)

    # AMP
    use_amp = cfg.bf16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---------- 恢复 ----------
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0)
        print(f"[resume] 从 {resume_from} 恢复，step={start_step}")

    # ---------- 训练循环 ----------
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_path = os.path.join(cfg.log_dir, "train.log")
    log_f = open(log_path, "a", encoding="utf-8")

    model.train()
    step = start_step
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    # 采样 prompt
    sample_prompts = ["你是谁？", "你好", "你喜欢什么", "讲个故事"]

    try:
        while step < cfg.max_steps:
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)

                # 前向 + 损失
                with torch.amp.autocast("cuda",
                                        enabled=use_amp and device.type == "cuda"):
                    out = model(x, collect_all_exits=True)
                    final_logits = out["final_logits"]
                    loss = F.cross_entropy(
                        final_logits.reshape(-1, final_logits.size(-1)),
                        y.reshape(-1), ignore_index=-100)

                    # 中间层辅助损失（每 3 层取一个）
                    for i, layer_logits in enumerate(out["exit_logits"]):
                        if i % 3 == 0 and i < cfg.n_layers - 1:
                            loss = loss + 0.1 * F.cross_entropy(
                                layer_logits.reshape(-1, layer_logits.size(-1)),
                                y.reshape(-1), ignore_index=-100)

                # 反向
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()

                step += 1
                running_loss += loss.item()
                running_count += 1

                # 日志
                if step % 50 == 0:
                    avg = running_loss / running_count
                    dt = time.time() - t0
                    lr = opt.param_groups[0]["lr"]
                    msg = (f"[step {step:>6}] loss={avg:.3f} "
                           f"lr={lr:.2e} elapsed={dt:.0f}s")
                    print(msg)
                    log_f.write(msg + "\n")
                    log_f.flush()
                    running_loss = 0.0
                    running_count = 0

                # 评估
                if step % cfg.eval_every == 0:
                    ppl = evaluate(model, val_dl, device, use_amp)
                    msg = f"[eval  step {step}] val_ppl={ppl:.2f}"
                    print(msg)
                    log_f.write(msg + "\n")
                    log_f.flush()

                    # 采样
                    print(f"[sample step {step}]")
                    for p in sample_prompts:
                        gen = sample(model, tok, p, device, max_new=20)
                        line = f"  '{p}' → '{gen}'"
                        print(line)
                        log_f.write(line + "\n")
                    log_f.flush()

                # 保存
                if step % cfg.save_every == 0:
                    ckpt_path = os.path.join(cfg.ckpt_dir, f"step_{step}.pt")
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(),
                        "config": cfg.__dict__,
                        "step": step,
                    }, ckpt_path)
                    print(f"[save] {ckpt_path}")

                    # 只保留最近 3 个
                    ckpts = sorted(
                        [f for f in os.listdir(cfg.ckpt_dir)
                         if f.startswith("step_") and f.endswith(".pt")],
                        key=lambda f: int(f.split("_")[1].split(".")[0]))
                    for old in ckpts[:-3]:
                        os.remove(os.path.join(cfg.ckpt_dir, old))

                if step >= cfg.max_steps:
                    break

    except KeyboardInterrupt:
        print("\n[中断] 用户停止训练")

    # ---------- 最终保存 ----------
    final_path = os.path.join(cfg.ckpt_dir, "final.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "config": cfg.__dict__,
        "step": step,
    }, final_path)
    print(f"[done] step={step}, 保存到 {final_path}")

    # 最终采样
    print("\n[final samples]")
    for p in sample_prompts:
        gen = sample(model, tok, p, device, max_new=40)
        print(f"  '{p}' → '{gen}'")

    log_f.close()


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复")
    parser.add_argument("--steps", type=int, default=None,
                        help="覆盖 cfg.max_steps")
    parser.add_argument("--batch", type=int, default=None,
                        help="覆盖 cfg.batch_size")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.lr is not None:
        cfg.lr = args.lr
    if args.device is not None:
        cfg.device = args.device

    print("=" * 60)
    print("HybridBrain M0 预训练")
    print("=" * 60)
    print(f"  模型：{cfg.n_layers} × {cfg.d_model} "
          f"(heads={cfg.n_heads}, kv={cfg.n_kv_heads})")
    print(f"  词表：{cfg.vocab_size}")
    print(f"  序列：{cfg.max_seq_len}")
    print(f"  批大小：{cfg.batch_size}")
    print(f"  学习率：{cfg.lr}")
    print(f"  最大步数：{cfg.max_steps}")
    print("=" * 60)

    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()