"""导出反馈候选供人工审核。"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve.memory.feedback import get_feedback_store

store = get_feedback_store()
stats = store.stats()
print("===== 反馈统计 =====")
for k, v in stats.items():
    print(f"  {k}: {v}")

cands = store.export_candidates(min_confidence=0.6, limit=500)
print(f"\n候选数: {len(cands)}")

os.makedirs("data/candidates", exist_ok=True)
out_path = "data/candidates/v10_candidates.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for c in cands:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"[ok] 已导出 → {out_path}")
print("\n===== 前 5 条预览 =====")
for c in cands[:5]:
    print(f"\n[{c['type']} conf={c['confidence']}]")
    for m in c["messages"]:
        print(f"  {m['role']}: {m['content'][:80]}")
