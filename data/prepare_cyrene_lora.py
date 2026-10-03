"""把 cyrene_clean.json 转成 LoRA 训练格式（chat template）。"""
import json
import os
import random

SRC = "/mnt/workspace/data/cyrene_clean.json"
OUT = "/mnt/workspace/data/cyrene_lora/train.jsonl"

SYSTEM = "你是昔涟，来自翁法罗斯的少女。性格温柔、喜欢听故事和看星星，语气轻盈。"

def main():
    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    print(f"[data] 原始 {len(samples)} 条")

    formatted = []
    for s in samples:
        q = s.get("input") or s.get("instruction") or ""
        a = s.get("output") or ""
        if not q or not a:
            continue
        formatted.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })

    # 划分 train / val
    random.seed(42)
    random.shuffle(formatted)
    n_val = max(10, len(formatted) // 10)
    val = formatted[:n_val]
    train = formatted[n_val:]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    val_path = OUT.replace("train", "val")
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[ok] train={len(train)} val={len(val)}")
    print(f"     {OUT}")
    print(f"     {val_path}")

if __name__ == "__main__":
    main()
