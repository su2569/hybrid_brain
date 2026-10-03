"""把 LoRA 权重合并到 base 模型，产出独立模型。"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = ("/mnt/workspace/models/models/"
              "Qwen--Qwen3-1.7B/snapshots/master")
LORA_PATH = "/mnt/workspace/checkpoints/qwen_cyrene_lora"
OUT_PATH = "/mnt/workspace/models/qwen3_cyrene_merged"


def main():
    print(f"[load base] {MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)

    print(f"[load lora] {LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.merge_and_unload()

    os.makedirs(OUT_PATH, exist_ok=True)
    model.save_pretrained(OUT_PATH)
    tok.save_pretrained(OUT_PATH)
    print(f"[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
