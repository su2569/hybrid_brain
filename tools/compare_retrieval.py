"""对比：无检索 vs 有检索的 LM loss。"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from tokenizer import CharTokenizer
from model.brain import Brain
from memory.knowledge import KnowledgeBase
from data.dataset import TextDataset, load_corpus


cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = CharTokenizer.from_vocab_json("data/all_chars_vocab.json")
cfg.vocab_size = len(tok)

brain = Brain(cfg).to(device)
ckpt = torch.load("/mnt/workspace/checkpoints/final.pt", map_location=device)
brain.backbone.load_state_dict(ckpt["model"], strict=False)

rckpt = torch.load("/mnt/workspace/checkpoints/m2_retriever_v5.pt",
                   map_location=device)
brain.retriever.load_state_dict(rckpt["retriever"])

with open("/mnt/workspace/checkpoints/m2_kb_v5.pkl", "rb") as f:
    kb_data = pickle.load(f)
kb = KnowledgeBase(dim=cfg.d_model)
kb.nodes = kb_data["nodes"]
brain.attach_kb(kb)

brain.eval()

# 验证集：从 LCCC 后段取
texts = load_corpus("data/lccc_50k.json", max_samples=25000,
                    assistant_name="昔涟")[13000:14000]
ds = TextDataset(texts, tok, cfg.max_seq_len)
dl = DataLoader(ds, batch_size=8)


@torch.no_grad()
def eval_mode(use_retriever):
    total, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        out = brain(x, need_heads=False, use_retriever=use_retriever)
        h = out["hidden"]
        logits = brain.backbone.exit_heads[-1](h)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1), ignore_index=-100)
        total += loss.item()
        n += 1
    return total / max(n, 1)


loss_no = eval_mode(False)
loss_ret = eval_mode(True)
print(f"无检索: loss={loss_no:.4f}")
print(f"有检索: loss={loss_ret:.4f}")
delta = (loss_no - loss_ret) / loss_no * 100
print(f"变化: {delta:+.2f}%  ({'✅ 有效' if delta > 1 else '❌ 无效'})")
