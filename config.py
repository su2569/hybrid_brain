"""全局配置。所有已确认的决策集中在这里。"""
from dataclasses import dataclass


@dataclass
class Config:
    # ===== 模型规模（M0 起点）=====
    vocab_size: int = 61050
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    n_kv_heads: int = 4
    max_seq_len: int = 512
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
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 100_000
    grad_clip: float = 1.0
    bf16: bool = True

    # ===== 数据 =====
    data_dir: str = "data"
    lccc_path: str = "LCCC/raw/LCCC-base_train.json"
    lccc_max_samples: int = 500_000
    cyrene_path: str = "data/cyrene_clean.json"
    cyrene_weight: float = 0.10
    val_ratio: float = 0.02

    # ===== 存储 =====
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every: int = 1000
    eval_every: int = 500

    # ===== 设备 =====
    device: str = "cuda"
    seed: int = 42


REL_TYPES = ("semantic", "knowledge", "information")