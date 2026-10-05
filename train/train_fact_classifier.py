"""BGE + LR 二分类：判断 query 是否是"陈述事实"。"""
import json, os, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BGE = "/mnt/workspace/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master"
TRAIN = "data/fact_classifier/train.jsonl"
VAL   = "data/fact_classifier/val.jsonl"
OUT   = "checkpoints/fact_classifier.pkl"

os.makedirs("checkpoints", exist_ok=True)

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

train = load(TRAIN)
val   = load(VAL)
print(f"train={len(train)} val={len(val)}")

print(">> 编码中...")
bge = SentenceTransformer(BGE)
X_tr = bge.encode([x["text"] for x in train], normalize_embeddings=True)
y_tr = np.array([x["label"] for x in train])
X_va = bge.encode([x["text"] for x in val], normalize_embeddings=True)
y_va = np.array([x["label"] for x in val])

print(">> 训练 LR...")
clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_va)
acc = accuracy_score(y_va, y_pred)
print(f"\nval accuracy: {acc:.4f}")
print(classification_report(y_va, y_pred, target_names=["neg", "pos"]))

with open(OUT, "wb") as f:
    pickle.dump({"clf": clf, "bge_path": BGE}, f)
print(f"[ok] saved: {OUT}")
