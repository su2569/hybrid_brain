"""意图检测 + 情感极性 + 行为预测（辅助）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


INTENT_NAMES = ("query", "chat", "abuse", "manipulate")
SENTIMENT_NAMES = ("positive", "negative", "neutral")
BEHAVIOR_NAMES = ("continue", "correct", "end", "switch")


class IntentAndPredictHead(nn.Module):
    def __init__(self, cfg: Config, dropout=0.3):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(dropout)
        self.shared = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        # 三个头
        self.intent_head = nn.Linear(cfg.d_model, cfg.n_intent_classes)
        self.sentiment_head = nn.Linear(cfg.d_model, 3)   # 情感
        self.behavior_head = nn.Linear(cfg.d_model, 4)    # 辅助

    def _pool(self, h, mask=None):
        if mask is None:
            return h.mean(dim=1)
        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp(min=1.0)
        return (h * m).sum(dim=1) / denom

    def forward(self, h, mask=None):
        feat = self._pool(h, mask)
        shared = self.shared(self.norm(feat))
        shared = self.dropout(shared)
        return {
            "intent_logits": self.intent_head(shared),
            "sentiment_logits": self.sentiment_head(shared),
            "behavior_logits": self.behavior_head(shared),
            "shared_feat": shared,
        }
