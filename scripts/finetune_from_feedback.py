"""用人工审核通过的候选微调。"""
import json, sys, os, random, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import PeftModel, LoraConfig, get_peft_model

BASE = "/dev/shm/hb_models/qwen3-1.7b"
PREV_LORA = "/mnt/workspace/checkpoints/qwen_cyrene_lora_v9"
NEW_LORA = "/mnt/workspace/checkpoints/qwen_cyrene_lora_v10"
CAND_PATH = "data/candidates/v10_candidates_accepted.jsonl"
SYS = "你是昔涟，来自翁法罗斯的少女。性格温柔、真诚，喜欢听故事和看星星。"

# 读候选
with open(CAND_PATH, encoding="utf-8") as f:
    cands = [json.loads(l) for l in f if l.strip()]

# 加 system
samples = []
for c in cands:
    msgs = [{"role": "system", "content": SYS}] + c["messages"]
    samples.append({"messages": msgs})

print(f"训练样本: {len(samples)}")
if len(samples) < 30:
    print("[warn] 样本 < 30，不推荐训练")
    sys.exit(1)

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def preprocess(x):
    msgs = x["messages"]
    full = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=False,
                                   enable_thinking=False)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=False)
    fi = tok(full, truncation=True, max_length=512,
             add_special_tokens=False)["input_ids"]
    pi = tok(prompt, truncation=True, max_length=512,
             add_special_tokens=False)["input_ids"]
    labels = list(fi)
    for j in range(min(len(pi), len(labels))):
        labels[j] = -100
    return {
        "input_ids": torch.tensor(fi, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

class DS(Dataset):
    def __init__(self, items): self.s = items
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return preprocess(self.s[i])

class Collator:
    def __init__(self, pad_id): self.pad_id = pad_id
    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        ids, labs = [], []
        for x in batch:
            p = max_len - len(x["input_ids"])
            ids.append(torch.cat([x["input_ids"], torch.full((p,), self.pad_id, dtype=torch.long)]))
            labs.append(torch.cat([x["labels"], torch.full((p,), -100, dtype=torch.long)]))
        return {"input_ids": torch.stack(ids), "labels": torch.stack(labs),
                "attention_mask": (torch.stack(ids) != self.pad_id).long()}

# 加载 base + 旧 LoRA
print(">> 加载 base...")
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, PREV_LORA, adapter_name="persona")
model = model.merge_and_unload()   # 合并旧 LoRA 到 base
model.config.use_cache = False

# 再挂新 LoRA
print(">> 挂新 LoRA...")
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=NEW_LORA,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=5e-5,   # 更小，避免破坏
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model, args=args, train_dataset=DS(samples),
    data_collator=Collator(tok.pad_token_id),
)
trainer.train()
trainer.save_model(NEW_LORA)
tok.save_pretrained(NEW_LORA)
print(f"[ok] 新 LoRA: {NEW_LORA}")
