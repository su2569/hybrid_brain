"""全局配置。所有已确认的决策集中在这里。"""
from dataclasses import dataclass


@dataclass
class Config:
    # ===== 模型规模（M0 起点）=====
    vocab_size: int = 61050
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2
    n_layers: int = 15
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 1000000.0
    rmsnorm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    # ===== MoE（暂放后 1/3 层）=====
    n_experts: int = 6
    top_k_experts: int = 2
    moe_start_layer: int = 8
    load_balance_weight: float = 0.01

    # ===== 特殊早退 =====
    early_exit_min_layer: int = 3
    early_exit_threshold: float = 0.9
    consistency_ratio: float = 0.95

    # ===== 意图检测 / 行为预测（M1）=====
    n_intent_classes: int = 4
    n_predict_tokens: int = 16

    # ===== 训练 =====
    batch_size: int = 64
    lr: float = 3e-6
    weight_decay: float = 0.01
    warmup_steps: int = 0
    max_steps: int = 100_000
    grad_clip: float = 0.5
    bf16: bool = True

    # ===== 数据 =====
    data_dir: str = "data"
    lccc_path: str = "data/lccc_50k.json"
    lccc_max_samples: int = 500_000
    cyrene_path: str = "data/cyrene_clean.json"
    cyrene_weight: float = 0.10
    val_ratio: float = 0.02

    # ===== 存储 =====
    ckpt_dir: str = "/mnt/workspace/checkpoints"
    log_dir: str = "/mnt/workspace/logs"
    save_every: int = 1000
    eval_every: int = 500

    # ===== 设备 =====
    device: str = "cuda"
    seed: int = 42


REL_TYPES = ("semantic", "knowledge", "information")