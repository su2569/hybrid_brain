"""测 NP-LoRA 融合 v9 persona + tool_v2。"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from peft.tuners.lora import LoraLayer

BASE         = "/mnt/workspace/models/models/Qwen--Qwen3-1.7B/snapshots/master"
PERSONA_LORA = "/mnt/workspace/checkpoints/qwen_cyrene_lora_v9"
TOOL_LORA    = "/mnt/workspace/checkpoints/qwen_tool_lora_v2"
MU = 0.5

PERSONA_SYS = ("你是昔涟，来自翁法罗斯的少女。"
               "性格温柔、诗意，喜欢听故事和看星星。"
               "语气轻盈，常用「呀」「呢」「♪」等语气词。")
TOOL_SYS = (
    "你是一个工具调用助手。根据用户需求判断是否需要：\n"
    "1. 追问澄清（<call type=\"clarify\">）\n"
    "2. 主动建议（<call type=\"suggest\">）\n"
    "3. 工具调用（<call type=\"action\" ...>）\n"
    "普通对话不要输出 <call>。"
)

PROMPTS = [
    ("tool", "帮我查一下明天上海的天气"),
    ("tool", "用 astrbot 搜一下量子计算"),
    ("tool", "codex 在 /tmp 执行 ls -la"),
    ("tool", "dsh 读一下 README.md"),
    ("tool", "算一下 1234*5678"),
    ("tool", "这款手机多少钱"),
    ("chat", "今天心情不好"),
    ("chat", "你是谁"),
]

print(">> 加载", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, PERSONA_LORA, adapter_name="persona")
model.load_adapter(TOOL_LORA, adapter_name="np_tool")

n_mod = 0
for name, module in model.named_modules():
    if not isinstance(module, LoraLayer): continue
    if "np_tool" not in module.lora_A or "persona" not in module.lora_A: continue
    B_s = module.lora_B["persona"].weight.data.float()
    B_c = module.lora_B["np_tool"].weight.data.float()
    if B_s.abs().max() < 1e-8 or B_c.abs().max() < 1e-8: continue
    U, S, Vh = torch.linalg.svd(B_s, full_matrices=False)
    proj = U @ (U.T @ B_c)
    B_c_new = B_c - (MU / (1.0 + MU)) * proj
    module.lora_B["np_tool"].weight.data.copy_(
        B_c_new.to(module.lora_B["np_tool"].weight.dtype))
    n_mod += 1
print(f">> NP-LoRA 投影: {n_mod} 层", flush=True)

model.add_weighted_adapter(
    adapters=["persona", "np_tool"], weights=[1.0, 1.0],
    adapter_name="pt", combination_type="cat")
model.set_adapter("pt")
model.eval()

def run(sys, prompt):
    msgs = [{"role":"system","content":sys},{"role":"user","content":prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="pt",
              add_special_tokens=False)["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=200, do_sample=False,
                             repetition_penalty=1.05)
    txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    has = "<call" in txt
    print(f"\n[{prompt}]  <call>={'YES' if has else 'NO'}")
    print(f"  {txt[:200]}")

print("\n===== 测试 =====", flush=True)
for kind, p in PROMPTS:
    run(TOOL_SYS if kind == "tool" else PERSONA_SYS, p)
