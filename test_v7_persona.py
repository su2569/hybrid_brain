import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "/mnt/workspace/models/models/Qwen--Qwen3-1.7B/snapshots/master"
LORA = "/mnt/workspace/checkpoints/qwen_cyrene_lora_v7_r32"

SYS = "你是昔涟，来自翁法罗斯的少女。性格温柔、喜欢听故事和看星星，语气轻盈。"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, LORA)
model.eval()

def run(prompt):
    msgs = [{"role":"system","content":SYS},{"role":"user","content":prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=150, do_sample=False,
                             repetition_penalty=1.1)
    print(f"\n>>> {prompt}")
    print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))

print("===== 训练集内 =====")
run("今天心情不好")
run("我睡不着")
run("刚失恋，心好痛")

print("\n===== 训练集外（泛化）=====")
run("我今天升职了")
run("想跟你说个秘密")
run("你觉得我该不该换工作")
run("刚收到拒信，好挫败")
