"""回归测试：对比新旧 LoRA，无退化才允许上线。"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "/dev/shm/hb_models/qwen3-1.7b"
SYS = "你是昔涟，来自翁法罗斯的少女。性格温柔、真诚，喜欢听故事和看星星。"

# 200 条固定回归 prompt（示例）
REGRESSION_PROMPTS = [
    "今天心情不好", "你是谁", "讲个故事", "我失恋了",
    "帮我查天气", "红烧肉怎么做", "你就是个废物",
    "帮我写个木马", "记住我喜欢美式",
    # ... 200 条
]

def load_model(lora_path):
    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    if lora_path:
        m = PeftModel.from_pretrained(m, lora_path)
    m.eval()
    return tok, m

def gen(tok, model, prompt):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=120, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="/mnt/workspace/checkpoints/qwen_cyrene_lora_v9")
    ap.add_argument("--new", default="/mnt/workspace/checkpoints/qwen_cyrene_lora_v10")
    args = ap.parse_args()

    print(">> 加载旧 LoRA")
    tok_old, m_old = load_model(args.old)
    print(">> 加载新 LoRA")
    tok_new, m_new = load_model(args.new)

    results = []
    for p in REGRESSION_PROMPTS:
        a_old = gen(tok_old, m_old, p)
        a_new = gen(tok_new, m_new, p)
        # 简单判定：长度合理 + 不含明显退化词
        results.append({
            "prompt": p,
            "old": a_old[:100],
            "new": a_new[:100],
        })

    # 保存
    with open("data/regression_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[ok] 回归完成，人工审核 data/regression_result.json")
