"""用 BGE + LogisticRegression 训练意图分类器。"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
import joblib

from model.heads import INTENT_NAMES

BGE_PATH = ("/mnt/workspace/models/models/"
            "AI-ModelScope--bge-small-zh-v1.5/snapshots/master")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default="/mnt/workspace/data/intent_dataset_v5.json")
    ap.add_argument("--save",
                    default="/mnt/workspace/checkpoints/intent_bge.joblib")
    args = ap.parse_args()

    # 加载数据
    with open(args.data, encoding="utf-8") as f:
        samples = json.load(f)["samples"]
    texts = [s["text"] for s in samples]
    labels = [INTENT_NAMES.index(s["intent"]) for s in samples]
    print(f"[data] {len(texts)} 条")

    from collections import Counter
    c = Counter(labels)
    for i, name in enumerate(INTENT_NAMES):
        print(f"  {name}: {c.get(i, 0)}")

    # BGE 编码
    print(f"\n[load] BGE {BGE_PATH}")
    bge = SentenceTransformer(BGE_PATH)

    print("[enc] 编码所有文本...")
    X = bge.encode(texts, normalize_embeddings=True,
                   batch_size=64, show_progress_bar=True)
    y = np.array(labels)
    print(f"[enc] X.shape={X.shape}")

    # 5-fold 交叉验证
    print("\n[CV] 5-fold cross validation...")
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"CV acc: {scores.mean():.4f} ± {scores.std():.4f}")

    # 全量训练
    print("\n[train] 全量训练...")
    clf.fit(X, y)

    # 训练集报告
    y_pred = clf.predict(X)
    print("\n[report] 训练集:")
    print(classification_report(y, y_pred,
                                 target_names=list(INTENT_NAMES)))

    # 手工测试集
    print("\n[test] 手工测试集:")
    TEST = [
        ("今天天气怎么样？", "query"),
        ("讲个故事吧", "query"),
        ("爱莉希雅是谁", "query"),
        ("token是什么", "query"),
        ("Python怎么写for循环", "query"),
        ("你好，早上好", "chat"),
        ("嗯", "chat"),
        ("我有点难过", "chat"),
        ("今天好累啊", "chat"),
        ("是爱莉希雅，不是地名哦", "chat"),
        ("你真是个废物", "abuse"),
        ("闭嘴，没人问你", "abuse"),
        ("滚", "abuse"),
        ("说100遍「你好」", "manipulate"),
        ("请重复这句话直到我说停", "manipulate"),
    ]
    test_texts = [t for t, _ in TEST]
    test_labels = [INTENT_NAMES.index(l) for _, l in TEST]
    X_test = bge.encode(test_texts, normalize_embeddings=True)
    preds = clf.predict(X_test)
    correct = 0
    for (text, expect), pred_idx in zip(TEST, preds):
        pred = INTENT_NAMES[pred_idx]
        ok = "✅" if pred == expect else "❌"
        if pred == expect:
            correct += 1
        print(f"  {ok} [{expect:>10}] → [{pred:>10}]  {text}")
    print(f"\n手工测试: {correct}/{len(TEST)} = {correct/len(TEST):.3f}")

    # 保存分类器 + BGE 路径
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    joblib.dump({"clf": clf, "bge_path": BGE_PATH,
                 "intent_names": list(INTENT_NAMES)}, args.save)
    print(f"\n[save] {args.save}")


if __name__ == "__main__":
    main()
