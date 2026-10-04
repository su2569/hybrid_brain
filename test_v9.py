import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "/mnt/workspace/models/models/Qwen--Qwen3-1.7B/snapshots/master"
LORA = "/mnt/workspace/checkpoints/qwen_cyrene_lora_v9"

POETIC_SYS = "你是昔涟，来自翁法罗斯的少女。性格温柔、喜欢听故事和看星星，语气轻盈。"
NON_SYS = "你是昔涟，用平实温柔的方式陪伴用户。"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, LORA)
model.eval()

def run(sys, p):
    msgs = [{"role":"system","content":sys},{"role":"user","content":p}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=200, do_sample=False, repetition_penalty=1.1)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

print("===== 诗类 system =====")
for p in ["今天心情不好", "刚失恋，心好痛", "你是谁？", "涟漪是什么？", "我坚持不下去了"]:
    print(f"\n>>> {p}\n{run(POETIC_SYS, p)}")

print("\n\n===== 非诗 system =====")
for p in ["代码跑不通，debug 了好久", "被老板骂了", "明天是妈妈的忌日", "牙疼得睡不着"]:
    print(f"\n>>> {p}\n{run(NON_SYS, p)}")

print("\n\n===== 泛化（诗类 system）=====")
for p in ["我今天升职了", "想跟你说个秘密", "刚收到拒信", "你相信命运吗"]:
    print(f"\n>>> {p}\n{run(POETIC_SYS, p)}")
