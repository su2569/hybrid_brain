"""从 DuReader 构建 QA 知识库。"""
import csv
import os
import sys
import time
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import Config
from tokenizer import CharTokenizer
from model.backbone import Backbone
from memory.knowledge import KnowledgeBase


def load_dureader(path="/mnt/data6/train.csv", max_n=20000):
    data = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            passage = row[0].strip()
            question = row[1].strip()
            passage = passage.split("[SEP]")[-1].strip()
            if 10 <= len(passage) <= 300:
                data.append({"passage": passage, "question": question})
            if len(data) >= max_n:
                break
    return data


@torch.no_grad()
def build_kb(model, tok, samples, cfg, device, batch_size=32):
    kb = KnowledgeBase(dim=cfg.d_model)
    model.eval()

    # 收集所有 passage
    passages = [s["passage"] for s in samples]
    print(f"[kb] passages: {len(passages)}")

    t0 = time.time()
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i+batch_size]
        ids_list = [tok.encode(p, add_bos=True, max_len=cfg.max_seq_len)
                    for p in batch]
        L = max(len(x) for x in ids_list)
        padded = torch.zeros(len(batch), L, dtype=torch.long)
        for j, ids in enumerate(ids_list):
            padded[j, :len(ids)] = torch.tensor(ids)
        padded = padded.to(device)

        out = model(padded)
        mask = (padded != 0).unsqueeze(-1).float()
        embs = (out["hidden"] * mask).sum(1) / mask.sum(1).clamp(min=1)

        for j, passage in enumerate(batch):
            nid = kb.add(passage, embs[j].cpu(),
                         space={"domain": "qa", "trust": 0.9},
                         status="active")
            kb.nodes[nid].hidden_tokens = out["hidden"][j, :len(ids_list[j])].cpu()

    print(f"[kb] 构建 {len(kb.nodes)} 条 ({time.time()-t0:.0f}s)")
    return kb


def main():
    cfg = Config()
    device = torch.device("cuda")
    tok = CharTokenizer.from_vocab_json("data/all_chars_vocab.json")
    cfg.vocab_size = len(tok)

    # 加载骨干
    model = Backbone(cfg).to(device)
    ckpt = torch.load("/mnt/workspace/checkpoints/final.pt", map_location=device)
    state = {k.replace("backbone.", ""): v for k, v in ckpt["model"].items()
             if k.startswith("backbone.")}
    model.load_state_dict(state if state else ckpt["model"], strict=False)
    print("[ckpt] backbone loaded")

    # 加载数据
    samples = load_dureader(max_n=10000)
    print(f"[data] {len(samples)} 条")

    # 构建 KB
    kb = build_kb(model, tok, samples, cfg, device)

    # 保存
    with open("/mnt/workspace/checkpoints/qa_kb.pkl", "wb") as f:
        pickle.dump({"nodes": kb.nodes, "version": kb.version}, f)
    print("[save] /mnt/workspace/checkpoints/qa_kb.pkl")


if __name__ == "__main__":
    main()
