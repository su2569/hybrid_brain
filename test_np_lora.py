"""
NP-LoRA: 把 tool LoRA 投影到 persona LoRA 风格子空间的零空间，
在保留 persona 语气的同时恢复 tool 调用能力。

对比 5 组:
  1. persona only
  2. tool only
  3. add_weighted linear
  4. add_weighted cat
  5. NP-LoRA (投影 + cat 合并)
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from peft.tuners.lora import LoraLayer

# ============ 配置 ============
BASE         = "/mnt/workspace/models/models/Qwen--Qwen3-1.7B/snapshots/master"
PERSONA_LORA = "/mnt/workspace/checkpoints/qwen_cyrene_lora_v6"
TOOL_LORA    = "/mnt/workspace/checkpoints/qwen_tool_lora"

MU = 0.5         # 投影强度: 0=不投影(等于合并), 1=半投影, 5=强投影
TOPK = None      # 取前 k 个奇异向量。None = 全部 r 个
# ==============================

# ============ system ============
PERSONA_SYS = "你是昔涟，来自翁法罗斯的少女。性格温柔、喜欢听故事和看星星，语气轻盈。"
TOOL_SYS = (
    "你是一个工具调用助手。根据用户需求判断是否需要：\n"
    "1. 追问澄清（<call type=\"clarify\">）\n"
    "2. 主动建议（<call type=\"suggest\">）\n"
    "3. 工具调用（<call type=\"action\" ...>）\n"
    "普通对话不要输出 <call>。"
)

# ============ prompts ============
PROMPTS = [
    ("tool", "这款手机多少钱？"),
    ("tool", "帮我查一下明天上海的天气"),
    ("tool", "算一下 1234*5678"),
    ("chat", "今天心情不好"),
    ("chat", "你叫什么名字"),
]

# ============ 加载 ============
print(">> 加载 base + 两个 LoRA")
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, PERSONA_LORA, adapter_name="persona")
model.load_adapter(TOOL_LORA, adapter_name="tool")
print(">> 复制一份 tool 作为 np_tool")
model.load_adapter(TOOL_LORA, adapter_name="np_tool")


# ============ NP-LoRA 投影 ============
def apply_np_lora(model, mu=0.5, topk=None):
    """
    遍历所有 LoRA 层，把 np_tool 的 lora_B 投影到 persona 的零空间。
    公式: B_c_new = B_c - μ/(1+μ) * P @ B_c
          P = U_k @ U_k.T  (U_k 是 persona lora_B 的左奇异向量前 k 个)
    """
    n_total, n_modified = 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if "np_tool" not in module.lora_A or "persona" not in module.lora_A:
            continue
        n_total += 1

        B_s = module.lora_B["persona"].weight.data.float()    # (out, r)
        B_c = module.lora_B["np_tool"].weight.data.float()    # (out, r)

        if B_s.abs().max() < 1e-8 or B_c.abs().max() < 1e-8:
            continue

        # SVD 提取 persona 的主方向
        U, S, Vh = torch.linalg.svd(B_s, full_matrices=False)
        # U: (out, r) — B_s 的列空间（风格子空间）
        k = U.shape[1] if topk is None else min(topk, U.shape[1])
        U_k = U[:, :k]                                        # (out, k)

        # 投影到 persona 列空间，再从 tool 里减去
        proj = U_k @ (U_k.T @ B_c)                            # (out, r)
        B_c_new = B_c - (mu / (1.0 + mu)) * proj

        module.lora_B["np_tool"].weight.data.copy_(
            B_c_new.to(module.lora_B["np_tool"].weight.dtype)
        )
        n_modified += 1

    print(f">> NP-LoRA 投影完成: {n_modified}/{n_total} 层 (mu={mu}, k={topk})")


apply_np_lora(model, mu=MU, topk=TOPK)


# ============ 合并: persona + np_tool ============
print(">> 合并 persona + np_tool (cat)")
try:
    model.add_weighted_adapter(
        adapters=["persona", "np_tool"],
        weights=[1.0, 1.0],
        adapter_name="np_merged",
        combination_type="cat",
    )
    print(">> ok")
except Exception as e:
    print(f">> fail: {e!r}")


# ============ 对照组合并 ============
print(">> 对照: linear 合并")
try:
    model.add_weighted_adapter(
        adapters=["persona", "tool"],
        weights=[1.0, 1.0],
        adapter_name="lin_merged",
        combination_type="linear",
    )
except Exception as e:
    print(f">> fail: {e!r}")

print(">> 对照: cat 合并")
try:
    model.add_weighted_adapter(
        adapters=["persona", "tool"],
        weights=[1.0, 1.0],
        adapter_name="cat_merged",
        combination_type="cat",
    )
except Exception as e:
    print(f">> fail: {e!r}")


model.eval()


# ============ 生成 ============
def run(adapter_name, sys, prompt):
    model.set_adapter(adapter_name)
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    ids = tok(text, return_tensors="pt",
              add_special_tokens=False)["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=160,
                             do_sample=False, repetition_penalty=1.05)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ============ 主循环 ============
CONFIGS = [
    ("persona",      "persona"),
    ("tool",         "tool"),
    ("linear合并",   "lin_merged"),
    ("cat合并",      "cat_merged"),
    ("NP-LoRA",      "np_merged"),
]

for label, adapter in CONFIGS:
    print("\n" + "=" * 70)
    print(f"【{label}】  adapter={adapter}")
    print("=" * 70)
    for kind, prompt in PROMPTS:
        sys = TOOL_SYS if kind == "tool" else PERSONA_SYS
        try:
            out = run(adapter, sys, prompt)
        except Exception as e:
            out = f"<ERROR> {e!r}"
        has_call = "<call" in out
        tag = "CALL" if has_call else "TEXT"
        print(f"\n[{kind}][{tag}] Q: {prompt}")
        # 只显示前 200 字符，避免刷屏
        print(f"A: {out[:200]}")

print("\n" + "=" * 70)
print("完成。判断标准:")
print("  - 3 个 tool 都该出现 <call>")
print("  - 2 个 chat 都该是昔涟语气且无 <call>")
print("  - NP-LoRA 应该在保持 persona 语气的同时恢复 <call>")
