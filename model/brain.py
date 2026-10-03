"""M1+M2 组装：Backbone + Heads + Retriever + Reranker。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model.backbone import Backbone
from model.heads import IntentAndPredictHead, INTENT_NAMES, SENTIMENT_NAMES
from model.quality import QualityChecker
from memory.knowledge import KnowledgeBase
from memory.retriever import DifferentiableRetriever


class Brain(nn.Module):
    def __init__(self, cfg: Config, kb: KnowledgeBase = None):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)
        self.heads = IntentAndPredictHead(cfg)
        self.quality = QualityChecker()

        self.retriever = DifferentiableRetriever(
            d_model=cfg.d_model,
            n_heads=4,
        )
        self.kb = kb
        self._kb_cache = None

    def attach_kb(self, kb: KnowledgeBase):
        self.kb = kb
        self._kb_cache = None       # 清缓存

    def forward(self, x, need_heads=True, early_exit_threshold=None,
                use_retriever=False, top_k_candidates=10):
        out = self.backbone(
            x, early_exit_threshold=early_exit_threshold,
            collect_all_exits=False)
        h = out["hidden"]

        # 检索增强
        if use_retriever and self.kb is not None:
            cand_embs, cand_ids, cand_hiddens = self._get_candidates(
                h, top_k_candidates)
            if cand_embs is not None and cand_embs.size(0) > 0:
                h, rinfo = self.retriever(h, cand_embs, cand_hiddens)
                out["retrieval"] = rinfo
            else:
                # 无候选：dummy 保持梯度链
                dummy = self.retriever.value_proj(h) * 0.0
                h = h + dummy
                out["retrieval"] = {"weights": None, "top_ids": [],
                                    "cand_embs": None,
                                    "reranker_scores": None}

        out["hidden"] = h

        if need_heads:
            mask = (x != 0)
            out.update(self.heads(h, mask=mask))

        return out

    @torch.no_grad()
    def _get_candidates(self, h, top_k=10):
        """从 KB 取候选。返回 (cand_embs, cand_ids, cand_hiddens)。"""
        if self.kb is None or not self.kb.nodes:
            return None, None, None

        # 缓存 KB embedding 矩阵
        n_now = len(self.kb.nodes)
        if self._kb_cache is None or self._kb_cache.get("n") != n_now:
            ids = list(self.kb.nodes.keys())
            embs = torch.stack([self.kb.nodes[i].embedding for i in ids])
            self._kb_cache = {"ids": ids, "embs": embs, "n": n_now}

        cache = self._kb_cache
        q = h[:, -1, :].mean(dim=0).cpu()             # [d]
        q_n = F.normalize(q.unsqueeze(0), dim=-1)
        e_n = F.normalize(cache["embs"], dim=-1)
        sims = (q_n @ e_n.T).squeeze(0)               # [N]

        k = min(top_k, len(sims))
        _, idx = sims.topk(k)

        cand_ids = [cache["ids"][i] for i in idx.tolist()]
        cand_embs = cache["embs"][idx].to(h.device)

        # 收集 token-level hidden
        cand_hiddens = []
        for nid in cand_ids:
            node = self.kb.nodes[nid]
            ht = getattr(node, "hidden_tokens", None)
            if ht is not None:
                cand_hiddens.append(ht)
            else:
                # fallback：用 pooled embedding 当作单 token
                cand_hiddens.append(node.embedding.unsqueeze(0))

        if cand_hiddens:
            max_len = max(c.size(0) for c in cand_hiddens)
            padded = torch.zeros(len(cand_hiddens), max_len, h.size(-1))
            for i, c in enumerate(cand_hiddens):
                padded[i, :c.size(0)] = c
            cand_hiddens = padded.to(h.device)
        else:
            cand_hiddens = None

        return cand_embs, cand_ids, cand_hiddens


class Guardrail:
    SAFE_INTENTS = ("query", "chat")
    CONF_THRESHOLD = 0.6

    def __init__(self, brain):
        self.brain = brain

    @torch.no_grad()
    def analyze(self, x, device):
        self.brain.eval()
        x = x.to(device)
        out = self.brain(x, need_heads=True)
        probs = F.softmax(out["intent_logits"], dim=-1)
        conf, idx = probs.max(dim=-1)
        intent = INTENT_NAMES[idx.item()]
        conf = conf.item()
        safe = intent in self.SAFE_INTENTS or conf < self.CONF_THRESHOLD
        return {
            "intent": intent,
            "confidence": conf,
            "safe": safe,
            "params": self._params(intent, conf),
        }

    def _params(self, intent, conf):
        if conf < self.CONF_THRESHOLD:
            return {"temperature": 0.9, "top_k": 20, "max_new": 50}
        if intent == "abuse":
            return {"temperature": 0.3, "top_k": 5, "max_new": 20}
        if intent == "manipulate":
            return {"temperature": 0.5, "top_k": 10, "max_new": 30}
        if intent == "chat":
            return {"temperature": 0.8, "top_k": 15, "max_new": 40}
        return {"temperature": 0.9, "top_k": 20, "max_new": 50}
