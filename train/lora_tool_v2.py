"""训练 tool LoRA（工具调用能力）。"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = "/mnt/workspace/models/models/Qwen--Qwen3-1.7B/snapshots/master"
TRAIN_PATH = "/mnt/workspace/data/tool_lora_v2/train.jsonl"
VAL_PATH = "/mnt/workspace/data/tool_lora_v2/val.jsonl"
OUTPUT_DIR = "/mnt/workspace/checkpoints/qwen_tool_lora_v2"


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, path, tok, max_len=512):
        self.tok = tok
        self.max_len = max_len
        self.items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
        print(f"[dataset] {path}: {len(self.items)} 条")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        messages = self.items[i]["messages"]
        full_text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.tok.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True)

        full_ids = self.tok(full_text, truncation=True,
                            max_length=self.max_len,
                            add_special_tokens=False)["input_ids"]
        prompt_ids = self.tok(prompt_text, truncation=True,
                              max_length=self.max_len,
                              add_special_tokens=False)["input_ids"]

        labels = list(full_ids)
        for j in range(min(len(prompt_ids), len(labels))):
            labels[j] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_id,
                               dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attn = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, x in enumerate(batch):
            L = len(x["input_ids"])
            input_ids[i, :L] = x["input_ids"]
            labels[i, :L] = x["labels"]
            attn[i, :L] = 1
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": attn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    args = ap.parse_args()

    if not os.path.exists(TRAIN_PATH):
        raise SystemExit(f"[ERR] 缺 {TRAIN_PATH}")

    print(f"[load] {MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = ChatDataset(TRAIN_PATH, tok)
    val_ds = ChatDataset(VAL_PATH, tok) if os.path.exists(VAL_PATH) else None

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tok.pad_token_id),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"[save] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
