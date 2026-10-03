"""四维知识库：节点 + 三类边 + 冲突仲裁 + 版本管理。"""
import copy
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
import torch.nn.functional as F


REL_TYPES = ("semantic", "knowledge", "information")


@dataclass
class Importance:
    score: float = 0.5
    use_count: int = 0
    conflict_win: int = 0


@dataclass
class KnowledgeNode:
    id: str
    content: str
    embedding: torch.Tensor
    space: dict = field(default_factory=dict)
    time: dict = field(default_factory=dict)
    importance: Importance = field(default_factory=Importance)
    relations: dict = field(default_factory=lambda: {t: [] for t in REL_TYPES})
    status: str = "candidate"
    embed_version: str = "v1"
    hidden_tokens: torch.Tensor = None   # [T_c, d] token-level hidden


def _new_edge(target, weight, expires_at=None):
    return {
        "target": target,
        "weight": float(weight),
        "direction": "out",
        "created": time.time(),
        "expires_at": expires_at,
        "status": "active",
    }


class KnowledgeBase:
    """四维知识库。

    节点：id + content + embedding + space + time + importance + relations
    边：semantic / knowledge / information（每类 ≤5，总数 ≤15）
    状态：candidate → active → superseded / conflict / dead
    """

    def __init__(self, dim: int, max_per_type: int = 5,
                 max_total: int = 15):
        self.dim = dim
        self.max_per_type = max_per_type
        self.max_total = max_total
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.snapshots: List[dict] = []
        self.audit = deque(maxlen=500)
        self.version = 0
        self._kseq = 0
        self._snapseq = 0

    # ---------- 日志 ----------
    def log(self, event: str, **data):
        self.audit.append({"time": time.time(), "event": event, **data})

    # ---------- ID ----------
    def _new_id(self) -> str:
        self._kseq += 1
        return f"k_{self._kseq}_{int(time.time()*1000)}"

    # ---------- 写入 ----------
    def add(self, content: str, embedding: torch.Tensor,
            space: dict = None, status: str = "candidate",
            trust: float = 0.5, embed_version: str = "v1") -> str:
        now = time.time()
        nid = self._new_id()

        sp = {"source": "unknown", "domain": None, "user": None,
              "modality": "text", "trust": trust}
        if space:
            sp.update(space)

        self.nodes[nid] = KnowledgeNode(
            id=nid,
            content=content,
            embedding=embedding.detach().cpu().flatten(),
            space=sp,
            time={"created": now, "last_used": now,
                  "expires_at": None, "version": 1},
            importance=Importance(score=0.5),
            status=status,
            embed_version=embed_version,
        )
        self.version += 1
        self.log("add", id=nid, status=status)
        return nid

    # ---------- 三类边 ----------
    def link(self, src_id: str, dst_id: str, rel_type: str,
             weight: float = 1.0, expires_at: float = None) -> bool:
        if rel_type not in REL_TYPES:
            return False
        src = self.nodes.get(src_id)
        if src is None or dst_id not in self.nodes:
            return False

        edges = src.relations[rel_type]

        # 去重
        if any(e["target"] == dst_id and e["status"] == "active"
               for e in edges):
            return True

        # 单类上限
        if len(edges) >= self.max_per_type:
            edges.sort(key=lambda e: e["weight"])
            edges.pop(0)

        # 总上限
        if sum(len(v) for v in src.relations.values()) >= self.max_total:
            lowest_t, lowest_e = None, None
            for t, es in src.relations.items():
                for e in es:
                    if lowest_e is None or e["weight"] < lowest_e["weight"]:
                        lowest_t, lowest_e = t, e
            if lowest_t is not None:
                src.relations[lowest_t].remove(lowest_e)

        edges.append(_new_edge(dst_id, weight, expires_at))
        self.version += 1
        return True

    # ---------- 查询 ----------
    def query(self, q: torch.Tensor, top_k: int = 5,
              filter_space: dict = None) -> List[dict]:
        """四步查询：语义召回 + 过滤 + 推理扩展 + 重排。"""
        q = q.detach().cpu().flatten()
        now = time.time()
        cands = {}

        for nid, node in self.nodes.items():
            if node.status in ("superseded", "dead"):
                continue

            # 时间过滤
            exp = node.time.get("expires_at")
            if exp and now > exp:
                node.status = "expired"
                continue

            # 空间过滤
            fs = filter_space or {}
            if fs.get("domain") and node.space.get("domain") != fs["domain"]:
                continue
            if fs.get("user") and node.space.get("user") not in (None, "global", fs["user"]):
                continue

            sim = F.cosine_similarity(q.unsqueeze(0),
                                      node.embedding.unsqueeze(0)).item()
            fresh = 1.0 / (1.0 + 1e-4 * (now - node.time.get("last_used", now)))
            trust = node.space.get("trust", 0.5)
            if node.status == "conflict":
                trust *= 0.7

            cands[nid] = {
                "sim": sim,
                "fresh": fresh,
                "trust": trust,
                "importance": node.importance.score,
                "know": 0.0,
                "info": 0.0,
            }

        if not cands:
            return []

        # 沿 knowledge/information 边扩展
        top = sorted(cands.items(), key=lambda kv: -kv[1]["sim"])[:max(top_k, 2)]
        for nid, _ in top:
            rels = self.nodes[nid].relations
            for e in rels.get("knowledge", []):
                if e["status"] == "active" and e["target"] in cands:
                    cands[e["target"]]["know"] = max(
                        cands[e["target"]]["know"], e["weight"] * 0.7)
            for e in rels.get("information", []):
                if e["status"] == "active" and e["target"] in cands:
                    cands[e["target"]]["info"] = max(
                        cands[e["target"]]["info"], e["weight"] * 0.7)

        # 重排
        W = {"sem": 0.30, "imp": 0.25, "time": 0.15,
             "space": 0.15, "know": 0.10, "info": 0.05}
        out = []
        for nid, c in cands.items():
            score = (W["sem"] * c["sim"] + W["imp"] * c["importance"]
                     + W["time"] * c["fresh"] + W["space"] * c["trust"]
                     + W["know"] * c["know"] + W["info"] * c["info"])
            out.append({"id": nid, "score": round(score, 4), **c})

        out.sort(key=lambda x: -x["score"])
        return out[:top_k]

    # ---------- 使用追踪 ----------
    def touch(self, nid: str):
        node = self.nodes.get(nid)
        if node:
            node.time["last_used"] = time.time()
            node.importance.use_count += 1

    # ---------- 冲突仲裁 ----------
    def detect_conflicts(self, embedding=None, domain=None,
                         node_id=None, content=None,
                         threshold: float = 0.85) -> List[dict]:
        if node_id:
            node = self.nodes.get(node_id)
            if node is None:
                return []
            emb = node.embedding
            dom = node.space.get("domain")
            content = node.content
        else:
            emb = embedding.detach().cpu().flatten()
            dom = domain

        out = []
        for nid, n in self.nodes.items():
            if node_id and nid == node_id:
                continue
            if n.status not in ("active", "candidate"):
                continue
            if dom and n.space.get("domain") != dom:
                continue
            sim = F.cosine_similarity(emb.unsqueeze(0),
                                      n.embedding.unsqueeze(0)).item()
            if sim >= threshold:
                out.append({"id": nid, "sim": round(sim, 4),
                            "content": n.content})
        out.sort(key=lambda x: -x["sim"])
        return out

    def arbitrate(self, new_id: str, old_id: str,
                  user_confirmed: bool = None) -> str:
        """三态仲裁：new_wins / old_wins / uncertain。"""
        new, old = self.nodes.get(new_id), self.nodes.get(old_id)
        if not new or not old or new_id == old_id:
            return "invalid"

        now = time.time()

        def score(n):
            trust = n.space.get("trust", 0.5)
            age = max(0.0, (now - n.time.get("created", now)) / 86400)
            return (0.4 * trust + 0.3 * (1.0 / (1.0 + age))
                    + 0.3 * n.importance.score)

        margin = score(new) - score(old)
        m = 0.05

        if user_confirmed is True:
            verdict = "new_wins"
        elif user_confirmed is False:
            verdict = "old_wins"
        elif margin > m:
            verdict = "new_wins"
        elif margin < -m:
            verdict = "old_wins"
        else:
            verdict = "uncertain"

        if verdict == "new_wins":
            new.importance.score = min(1.0, new.importance.score + 0.1)
            new.importance.conflict_win += 1
            new.status = "active"
            old.importance.score *= 0.5
            old.status = "superseded"
            self.link(new_id, old_id, "knowledge", 0.9)
        elif verdict == "old_wins":
            old.status = "active"      # ← 新增：旧节点提升为 active
            old.importance.score = min(1.0, old.importance.score + 0.05)
            old.importance.conflict_win += 1
            new.status = "candidate"
            new.importance.score *= 0.7
            self.link(old_id, new_id, "knowledge", 0.5)
        else:
            new.status = "conflict"
            old.status = "conflict"
            new.importance.score *= 0.7
            old.importance.score *= 0.7
            self.link(new_id, old_id, "knowledge", 0.3)
            self.link(old_id, new_id, "knowledge", 0.3)

        self.log("arbitrate", new=new_id, old=old_id, verdict=verdict,
                 margin=round(margin, 4))
        return verdict

    # ---------- 快照 ----------
    def snapshot(self) -> int:
        self._snapseq += 1
        sid = self._snapseq
        self.snapshots.append({
            "sid": sid,
            "time": time.time(),
            "version": self.version,
            "nodes": copy.deepcopy(self.nodes),
        })
        while len(self.snapshots) > 10:
            self.snapshots.pop(0)
        self.log("snapshot", sid=sid)
        return sid

    def rollback(self, sid: int) -> bool:
        rec = next((s for s in self.snapshots if s["sid"] == sid), None)
        if rec is None:
            return False
        self.nodes = copy.deepcopy(rec["nodes"])
        self.version = rec["version"]
        self.log("rollback", sid=sid)
        return True

    # ---------- 统计 ----------
    def stats(self) -> dict:
        by_status = {}
        for n in self.nodes.values():
            by_status[n.status] = by_status.get(n.status, 0) + 1
        return {
            "total": len(self.nodes),
            "by_status": by_status,
            "version": self.version,
        }
