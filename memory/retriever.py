"""可微检索器 v5：融合 reranker prior。"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from memory.reranker import Reranker


class DifferentiableRetriever(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4,
                 gate_max: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.gate_max = gate_max

        # attention 投影
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Reranker（作为 prior）
        self.reranker = Reranker(d_model, n_heads)
        # prior 强度，初始 1.0
        self.reranker_weight = nn.Parameter(torch.tensor(1.0))

        # gate
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, h, cand_embs, cand_hiddens=None, top_k=10):
        """
        h: [B, T, d]
        cand_embs: [K, d]
        cand_hiddens: [K, T_c, d]
        """
        if cand_embs is None or cand_embs.size(0) == 0:
            return h, {"weights": None, "top_ids": [],
                       "cand_embs": None, "reranker_scores": None,
                       "gate": 0.0}

        cand_embs = cand_embs.detach()
        B, T, D = h.shape
        K = cand_embs.size(0)

        # ---- 1) attention scores ----
        q = self.query_proj(h).view(B, T, self.n_heads, self.head_dim)
        k = self.key_proj(cand_embs).view(K, self.n_heads, self.head_dim)
        v = self.value_proj(cand_embs)             # [K, d]

        q = q.permute(0, 2, 1, 3)                   # [B, H, T, Dh]
        k = k.permute(1, 0, 2)                      # [H, K, Dh]
        attn_scores = torch.einsum("bhtd,hkd->bhtk", q, k) \
                      / math.sqrt(self.head_dim)     # [B, H, T, K]

        # ---- 2) reranker prior ----
        reranker_scores = None
        if cand_hiddens is not None and K > 0:
            reranker_scores = self.reranker(h, cand_hiddens)  # [B, K]
            # 加到 attention 分数上（广播到 H 和 T）
            attn_scores = attn_scores \
                + self.reranker_weight * reranker_scores[:, None, None, :]

        # ---- 3) 归一化 ----
        weights = F.softmax(attn_scores / 0.5, dim=-1)   # [B, H, T, K]
        weights = weights.mean(dim=1)                     # [B, T, K]

        # ---- 4) 加权组合 ----
        retrieved = torch.einsum("btk,kd->btd", weights, v)  # [B, T, d]

        # ---- 5) gate ----
        gate = torch.sigmoid(self.gate_logit) * self.gate_max
        h_enhanced = h + gate * retrieved

        return h_enhanced, {
            "weights": weights.detach(),
            "gate": float(gate),
            "cand_embs": cand_embs,
            "reranker_scores": reranker_scores.detach()
                if reranker_scores is not None else None,
        }
