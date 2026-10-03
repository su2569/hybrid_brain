"""可微检索器训练 v2。

关键改动：
  1. KB / 训练 / 验证严格隔离
  2. KB 只存知识型句子
  3. gate 上限 0.3
  4. 只在低置信位置注入
  5. 对比学习辅助损失
"""
import os
import sys
import argparse
import time
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from tokenizer import CharTokenizer
from model.brain import Brain
from memory.knowledge import KnowledgeBase
from data.dataset import TextDataset, load_corpus


def is_knowledge(text, min_len=10, max_len=100):
    """已经去掉了'用户/助手'前缀的句子。"""
    return min_len <= len(text) <= max_len


def extract_sentences(text):
    """从'用户：xxx\n助手：yyy'提取每句话。"""
    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 去前缀
        for prefix in ("用户：", "助手：", "昔涟："):
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        if line:
            sentences.append(line)
    return sentences


@torch.no_grad()
def build_kb_from_corpus(brain, texts, tok, cfg, device,
                         max_items=3000, batch_size=32):
    kb = KnowledgeBase(dim=cfg.d_model)
    brain.eval()

    # 从多轮文本提取单句
    candidates = []
    for t in texts:
        for s in extract_sentences(t):
            if is_knowledge(s):
                candidates.append(s)
                if len(candidates) >= max_items:
                    break
        if len(candidates) >= max_items:
            break

    filtered = candidates
    print(f"[kb] 知识型句子 {len(filtered)} 条")

    t0 = time.time()
    n_added = 0
    for i in range(0, len(filtered), batch_size):
        batch = filtered[i:i + batch_size]
        ids_list = [tok.encode(t, add_bos=True, max_len=cfg.max_seq_len)
                    for t in batch]
        max_len_b = max(len(x) for x in ids_list)
        padded = torch.zeros(len(batch), max_len_b, dtype=torch.long)
        for j, ids in enumerate(ids_list):
            padded[j, :len(ids)] = torch.tensor(ids)
        padded = padded.to(device)

        out = brain.backbone(padded)
        mask = (padded != 0).unsqueeze(-1).float()
        embs = (out["hidden"] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        for j, text in enumerate(batch):
            nid = kb.add(text, embs[j].cpu(), status="active", trust=0.8)
            # 保存 token-level hidden
            kb.nodes[nid].hidden_tokens = out["hidden"][j, :padded.size(1)].cpu()
            n_added += 1

    print(f"[kb] 构建 {n_added} 条 ({time.time()-t0:.0f}s)")
    return kb


def train_retriever(brain, train_texts, tok, cfg, device,
                    n_epochs=5, lr=1e-4, top_k_cand=10):
    for p in brain.backbone.parameters():
        p.requires_grad = False
    for p in brain.heads.parameters():
        p.requires_grad = False
    for p in brain.retriever.parameters():
        p.requires_grad = True
    # 冻结 reranker（已经训练好）
    for p in brain.retriever.reranker.parameters():
        p.requires_grad = False

    n_train = sum(p.numel() for p in brain.retriever.parameters())
    print(f"[trainable] retriever: {n_train/1e6:.2f}M")

    ds = TextDataset(train_texts, tok, cfg.max_seq_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=4, drop_last=True)

    opt = torch.optim.AdamW(brain.retriever.parameters(), lr=lr)
    brain.train()
    brain.backbone.eval()
    brain.heads.eval()

    step = 0
    t0 = time.time()
    running_loss = 0.0
    running_n = 0

    for epoch in range(n_epochs):
        for x, y in dl:
            x, y = x.to(device), y.to(device)

            out = brain(x, need_heads=True, use_retriever=True,
                        top_k_candidates=top_k_cand)

            # 主 loss
            h = out["hidden"]
            logits = brain.backbone.exit_heads[-1](h)
            loss_lm = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1), ignore_index=-100)

            # 对比学习辅助损失
            rinfo = out.get("retrieval", {})
            cand_embs = rinfo.get("cand_embs")
            aux_loss = torch.tensor(0.0, device=device)
            if cand_embs is not None and y.size(1) > 1:
                target_ids = y[:, 1:].clone()
                valid = (target_ids >= 0)
                target_ids = target_ids.clamp(min=0)
                target_emb = brain.backbone.embed(target_ids)
                # mean over valid
                target_emb = (target_emb * valid.unsqueeze(-1).float()).sum(1)
                target_emb = target_emb / valid.sum(1, keepdim=True).clamp(min=1).float()

                # 检索结果的加权表示
                weights = rinfo["weights"]                    # [B, T, K]
                # 用最后一个位置作为 query
                w_last = weights[:, -1, :]                    # [B, K]
                retrieved = w_last @ brain.retriever.value_proj(cand_embs)

                # 正样本：target_emb
                # 负样本：随机
                pos_sim = F.cosine_similarity(retrieved, target_emb, dim=-1)
                # 负样本：随机噪声 embedding（和候选不相关）
                neg_emb = torch.randn_like(retrieved)
                neg_sim = F.cosine_similarity(retrieved, neg_emb, dim=-1)

                aux_loss = -F.logsigmoid(pos_sim - neg_sim).mean()

            loss = loss_lm + 0.3 * aux_loss

            if not loss.requires_grad:
                raise RuntimeError("loss 无梯度")

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                brain.retriever.parameters(), 1.0)
            opt.step()

            step += 1
            running_loss += loss.item()
            running_n += 1

            if step % 50 == 0:
                avg = running_loss / running_n
                dt = time.time() - t0
                print(f"[step {step:>5}] loss={avg:.3f} "
                      f"aux={aux_loss.item():.3f} elapsed={dt:.0f}s")
                running_loss = 0.0
                running_n = 0

        print(f"[epoch {epoch}] done, step={step}")

    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/mnt/workspace/checkpoints/final.pt")
    parser.add_argument("--heads", default="/mnt/workspace/checkpoints/m1_heads_v7.pt")
    parser.add_argument("--save", default="/mnt/workspace/checkpoints/m2_retriever_v3.pt")
    parser.add_argument("--kb-save", default="/mnt/workspace/checkpoints/m2_kb_v3.pkl")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(args.device)
    print(f"[device] {device}")

    tok = CharTokenizer.from_vocab_json("data/all_chars_vocab.json")
    cfg.vocab_size = len(tok)

    brain = Brain(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    backbone_state = {
        k.replace("backbone.", ""): v
        for k, v in ckpt["model"].items() if k.startswith("backbone.")
    }
    if not backbone_state:
        backbone_state = ckpt["model"]
    brain.backbone.load_state_dict(backbone_state, strict=False)
    print(f"[ckpt] backbone")

    if os.path.exists(args.heads):
        hckpt = torch.load(args.heads, map_location=device)
        if "heads" in hckpt:
            brain.heads.load_state_dict(hckpt["heads"], strict=False)
            print(f"[ckpt] heads")

    # 加载已训练的 reranker
    reranker_path = "/mnt/workspace/checkpoints/m2_reranker.pt"
    if os.path.exists(reranker_path):
        rckpt = torch.load(reranker_path, map_location=device)
        if "reranker" in rckpt:
            brain.retriever.reranker.load_state_dict(rckpt["reranker"])
            print(f"[ckpt] reranker loaded from {reranker_path}")
        else:
            print(f"[warn] reranker ckpt 格式不对")

    # 加载 LCCC
    print(f"\n[data] 加载 LCCC...")
    all_texts = load_corpus("data/lccc_50k.json", max_samples=15000,
                            assistant_name="昔涟")
    print(f"[data] 总 {len(all_texts)} 条")

    # ===== 严格隔离 =====
    kb_texts = all_texts[:3000]
    train_texts = all_texts[3000:13000]
    val_texts = all_texts[13000:14000]
    print(f"[split] KB={len(kb_texts)} train={len(train_texts)} val={len(val_texts)}")

    # 构建 KB
    kb = build_kb_from_corpus(brain, kb_texts, tok, cfg, device, max_items=3000)
    brain.attach_kb(kb)

    # 训练
    print(f"\n[train] retriever")
    train_retriever(brain, train_texts, tok, cfg, device, n_epochs=args.epochs)

    # 保存
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({"retriever": brain.retriever.state_dict()}, args.save)
    with open(args.kb_save, "wb") as f:
        pickle.dump({"nodes": kb.nodes, "version": kb.version}, f)
    print(f"[save] {args.save} + {args.kb_save}")


if __name__ == "__main__":
    main()
