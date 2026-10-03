"""小型重排序网络：Transformer + CNN 混合。

v2 修复：
  - 所有池化走 mask，padding 不参与
  - cross_attn 加 key_padding_mask
  - CNN 前把 query 的 padding 置零
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Reranker(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4,
                 cnn_kernel: int = 3, hidden: int = 256):
        super().__init__()
        self.d_model = d_model

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(d_model)

        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2,
                      kernel_size=cnn_kernel, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model // 2,
                      kernel_size=cnn_kernel, padding=1),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Linear(d_model + d_model // 2 + d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _masked_mean(x, mask):
        """x: [*, T, d], mask: [*, T] bool (True=有效)"""
        if mask is None:
            return x.mean(dim=-2)
        m = mask.float().unsqueeze(-1)
        return (x * m).sum(dim=-2) / m.sum(dim=-2).clamp(min=1.0)

    def forward(self, q_hidden, c_hidden, q_mask=None, c_mask=None):
        """
        q_hidden: [B, T_q, d]
        c_hidden: [K, T_c, d]
        q_mask:   [B, T_q] bool, True=有效
        c_mask:   [K, T_c] bool, True=有效
        返回 scores: [B, K]
        """
        K, T_c, d = c_hidden.shape
        B, T_q, _ = q_hidden.shape

        # ---- 1) Cross-Attention：候选 attend query ----
        attn_out_list = []
        for b in range(B):
            q = c_hidden                         # [K, T_c, d]
            kv = q_hidden[b:b+1]                 # [1, T_q, d]
            kv_expand = kv.expand(K, -1, -1)     # [K, T_q, d]
            if q_mask is not None:
                # PyTorch 约定：True=屏蔽
                kpm = (~q_mask[b:b+1]).expand(K, -1)
            else:
                kpm = None
            out, _ = self.cross_attn(
                q, kv_expand, kv_expand, key_padding_mask=kpm)
            out = self.attn_norm(out + q)
            attn_out_list.append(out)

        attn_out = torch.stack(attn_out_list, dim=0)  # [B, K, T_c, d]

        if c_mask is not None:
            cm = c_mask.unsqueeze(0).expand(B, -1, -1).float().unsqueeze(-1)
            pooled_attn = (attn_out * cm).sum(2) / cm.sum(2).clamp(min=1.0)
        else:
            pooled_attn = attn_out.mean(dim=2)          # [B, K, d]

        # ---- 2) CNN：query 的 n-gram ----
        if q_mask is not None:
            q_in = q_hidden * q_mask.float().unsqueeze(-1)
        else:
            q_in = q_hidden
        q_trans = q_in.transpose(1, 2)                  # [B, d, T_q]
        cnn_feat = self.cnn(q_trans)                    # [B, d/2, T_q]
        if q_mask is not None:
            cm2 = q_mask.float().unsqueeze(1)           # [B, 1, T_q]
            cnn_feat = ((cnn_feat * cm2).sum(-1)
                        / cm2.sum(-1).clamp(min=1.0))
        else:
            cnn_feat = cnn_feat.mean(-1)                # [B, d/2]

        # ---- 3) 融合 ----
        q_pool = self._masked_mean(q_hidden, q_mask)    # [B, d]
        q_exp = q_pool.unsqueeze(1).expand(-1, K, -1)
        cnn_exp = cnn_feat.unsqueeze(1).expand(-1, K, -1)

        fused = torch.cat([q_exp, cnn_exp, pooled_attn], dim=-1)
        scores = self.fuse(fused).squeeze(-1)           # [B, K]
        return scores
