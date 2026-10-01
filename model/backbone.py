"""语言主干：12×768，Qwen 风格，含特殊早退。"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class Attention(nn.Module):
    """GQA + RoPE + 因果掩码"""
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        d_kv = self.n_kv * self.head_dim

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, d_kv, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, d_kv, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        inv_freq = 1.0 / (cfg.rope_theta ** (
            torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pad_mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        pos = torch.arange(T, device=x.device)
        freqs = torch.outer(pos.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(causal, float("-inf"))
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = ((int(cfg.d_model * 8 / 3) + 63) // 64) * 64
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.attn_norm(x), pad_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Backbone(nn.Module):
    """12×768 语言主干 + 特殊早退。

    特殊早退：
      - 标准早退：置信度够 → 返回，停止
      - 特殊早退：置信度够 → 记录早退结果，继续跑
        后续层做三件事：内省、意图检测、行为预测
      - 训练时每层 logits 都参与损失
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)

        # 每层早退头（也用于训练每层的中间监督）
        self.exit_heads = nn.ModuleList([
            nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            for _ in range(cfg.n_layers)
        ])

    def forward(self, x, early_exit_threshold=None,
                collect_all_exits=False, pad_mask=None):
        """
        返回 dict：
          - hidden: 最终 hidden
          - exit_logits: 每层输出（list，训练用）
          - early_exit: { layer, logits } 或 None
          - used_layers: 实际使用的层数
        """
        B, T = x.shape
        if pad_mask is None:
            pad_mask = (x == 0)
        valid = ~pad_mask

        h = self.embed(x)
        exit_logits_all = []
        early_exit = None
        used_layers = self.cfg.n_layers

        for i in range(self.cfg.n_layers):
            h = self.layers[i](h, pad_mask)
            layer_logits = self.exit_heads[i](h)

            if collect_all_exits:
                exit_logits_all.append(layer_logits)

            # 特殊早退：记录但不停
            if (early_exit is None
                    and early_exit_threshold is not None
                    and i >= self.cfg.early_exit_min_layer
                    and i < self.cfg.n_layers - 1):
                probs = F.softmax(layer_logits, dim=-1)
                conf, _ = probs.max(dim=-1)
                if bool(valid.any()):
                    conf_v = conf.masked_fill(~valid, 1.0)
                    if float(conf_v.min()) > early_exit_threshold:
                        early_exit = {"layer": i, "logits": layer_logits}

        h = self.norm_f(h)
        final_logits = self.exit_heads[-1](h)

        return {
            "hidden": h,
            "final_logits": final_logits,
            "exit_logits": exit_logits_all,
            "early_exit": early_exit,
            "used_layers": used_layers,
        }