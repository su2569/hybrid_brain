"""Qwen2.5-0.5B 专用配置。不影响原 Config。"""
from dataclasses import dataclass


@dataclass
class ConfigQwen:
    # ===== 骨干（Qwen2.5-0.5B）=====
    qwen_path: str = "/mnt/workspace/models/models/Qwen--Qwen3-1.7B/snapshots/master"
    d_model: int = 2048
    vocab_size: int = 151936
    n_layers: int = 28
    max_seq_len: int = 512       # Qwen 支持 32k，但内存友好取 512
    freeze_backbone: bool = True

    # ===== 三头 =====
    n_intent_classes: int = 4
    head_dropout: float = 0.3

    # ===== Reranker =====
    reranker_heads: int = 4
    reranker_hidden: int = 256

    # ===== 训练 =====
    batch_size: int = 16
    lr_heads: float = 5e-4
    lr_reranker: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    bf16: bool = True

    # ===== 数据 =====
    data_dir: str = "data"
    ckpt_dir: str = "/mnt/workspace/checkpoints"
    log_dir: str = "/mnt/workspace/logs/qwen"

    device: str = "cuda"
    seed: int = 42
