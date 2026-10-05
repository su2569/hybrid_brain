import json, pickle, os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BGE = "/mnt/workspace/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master"
os.makedirs("checkpoints", exist_ok=True)

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

train = load("data/classifiers/user_ref_train.jsonl")
val   = load("data/classifiers/user_ref_val.jsonl")
print(f"train={len(train)} val={len(val)}")

bge = SentenceTransformer(BGE)
X_tr = bge.encode([x["text"] for x in train], normalize_embeddings=True)
y_tr = np.array([x["label"] for x in train])
X_va = bge.encode([x["text"] for x in val], normalize_embeddings=True)
y_va = np.array([x["label"] for x in val])

clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_va)
print(f"val accuracy: {accuracy_score(y_va, y_pred):.4f}")
print(classification_report(y_va, y_pred))

with open("checkpoints/user_ref_classifier.pkl", "wb") as f:
    pickle.dump({"clf": clf, "bge_path": BGE,
                 "classes": list(clf.classes_)}, f)
print("[ok] checkpoints/user_ref_classifier.pkl")
