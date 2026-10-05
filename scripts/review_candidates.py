"""交互式人工审核候选。"""
import json, sys, os

path = sys.argv[1] if len(sys.argv) > 1 else "data/candidates/v10_candidates.jsonl"
accepted = []
rejected = []

with open(path, encoding="utf-8") as f:
    cands = [json.loads(l) for l in f if l.strip()]

print(f"共 {len(cands)} 条待审核")
print("操作: y=通过 n=拒绝 s=跳过 q=退出\n")

for i, c in enumerate(cands):
    print(f"\n===== [{i+1}/{len(cands)}] {c['type']} conf={c['confidence']:.2f} =====")
    for m in c["messages"]:
        print(f"{m['role']}: {m['content'][:200]}")
    try:
        cmd = input("y/n/s/q: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        break
    if cmd == "y":
        accepted.append(c)
    elif cmd == "n":
        rejected.append(c)
    elif cmd == "q":
        break

# 保存
acc_path = path.replace(".jsonl", "_accepted.jsonl")
rej_path = path.replace(".jsonl", "_rejected.jsonl")
with open(acc_path, "w", encoding="utf-8") as f:
    for c in accepted:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
with open(rej_path, "w", encoding="utf-8") as f:
    for c in rejected:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"\n通过: {len(accepted)} → {acc_path}")
print(f"拒绝: {len(rejected)} → {rej_path}")
