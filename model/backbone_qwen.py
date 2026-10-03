"""Qwen2.5-0.5B 包装为 HybridBrain 的 backbone。

接口对齐 model/backbone.py 的 Backbone：
  forward(x) -> {"hidden", "final_logits", "exit_logits", "early_exit", "used_layers"}

关键设计：
  - Qwen 全部参数冻结（后续只训 LoRA 或下游头）
  - 输入是 BPE token id（不是字符级）
  - 输出 hidden_states 兼容 d_model=896 的下游
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class QwenBackbone(nn.Module):
    def __init__(self, model_path="/mnt/workspace/models/Qwen2.5-0.5B",
                 freeze=True, dtype=torch.bfloat16):
        super().__init__()
        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype)
        self.cfg_d_model = self.qwen.config.hidden_size       # 896
        self.cfg_vocab_size = self.qwen.config.vocab_size     # 151936
        self.n_layers = self.qwen.config.num_hidden_layers    # 24

        if freeze:
            for p in self.qwen.parameters():
                p.requires_grad = False

        # 兼容老代码：有些地方会读 model.cfg
        class _Cfg:
            d_model = self.cfg_d_model
            vocab_size = self.cfg_vocab_size
            n_layers = self.n_layers
        self.cfg = _Cfg()

    def forward(self, x, early_exit_threshold=None,
                collect_all_exits=False, pad_mask=None):
        out = self.qwen(
            input_ids=x,
            output_hidden_states=True,
            return_dict=True,
        )
        h = out.hidden_states[-1]           # [B, T, 896]
        logits = out.logits                 # [B, T, 151936]

        return {
            "hidden": h,
            "final_logits": logits,
            "exit_logits": [],              # Qwen 无中间 exit，返回空
            "early_exit": None,
            "used_layers": self.n_layers,
        }

    def embed(self, x):
        """兼容某些代码调 embed"""
        return self.qwen.model.embed_tokens(x)
