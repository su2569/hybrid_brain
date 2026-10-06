"""集中管理所有模型/数据路径。唯一真相源。"""
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
HB_MODELS = "/mnt/workspace/hb_models"

MODELS = {
    "qwen3":        f"{HB_MODELS}/base/qwen3-1.7b",
    "bge_small":    f"{HB_MODELS}/base/bge-small-zh",
    "bge_reranker": f"{HB_MODELS}/base/bge-reranker-base",
}

CKPTS = {
    "persona":    f"{HB_MODELS}/lora/persona",
    "tool":       f"{HB_MODELS}/lora/tool",
    "route_clf":  f"{HB_MODELS}/clf/route_clf.pkl",
    "user_ref":   f"{HB_MODELS}/clf/user_ref_classifier.pkl",
    "fact_clf":   f"{HB_MODELS}/clf/fact_classifier.pkl",
    "intent_bge": f"{HB_MODELS}/clf/intent_bge.joblib",
    "heads_qwen": f"{HB_MODELS}/clf/heads_qwen.pt",
}

DATA = {
    "kb":        f"{_ROOT}/data/kb_merged.json",
    "kb_embs":   f"{_ROOT}/data/kb_embs_da0352ce685a.npy",
    "memory":    f"{_ROOT}/data/memory.db",
    "user_kb":   f"{_ROOT}/data/user_kb.db",
    "cyrene_v3": f"{_ROOT}/data/cyrene_lora_v3",
    "tool_v2":   f"{_ROOT}/data/tool_lora_v2",
}

LOGS = "/mnt/workspace/logs"
BACKUP = "/mnt/data"

if __name__ == "__main__":
    for name, d in [("MODELS", MODELS), ("CKPTS", CKPTS), ("DATA", DATA)]:
        print(f"=== {name} ===")
        for k, v in d.items():
            print(f"  {'✅' if os.path.exists(v) else '❌'} {k}: {v}")
