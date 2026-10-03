"""从 DuReader 构建 QA 知识库。

当前只需要 load_dureader（读 JSON）。其他旧函数（build_kb）已废弃。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dureader(path="/mnt/workspace/data6/dureader_robust-data/train.json",
                  max_n=20000):
    """读 DuReader robust JSON 格式。

    结构：
        {"data": [{"title": ..., "paragraphs": [
            {"context": "passage", "qas": [{"question": ..., "answers": [...]}]}
        ]}]}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = []
    for article in raw.get("data", []):
        for para in article.get("paragraphs", []):
            passage = para.get("context", "").strip()
            if not (10 <= len(passage) <= 500):
                continue
            for qa in para.get("qas", []):
                question = qa.get("question", "").strip()
                if question:
                    data.append({"passage": passage, "question": question})
                    if len(data) >= max_n:
                        return data
    return data
