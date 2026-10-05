#!/bin/bash
# 把模型复制到 /dev/shm 加速加载（容器重启后重跑）
set -e
SHM=/dev/shm/hb_models
mkdir -p "$SHM"

copy() {
    src="$1"; dst="$SHM/$2"
    if [ ! -d "$dst" ]; then
        echo ">> $2..."
        cp -r "$src" "$dst"
    else
        echo "[skip] $2 已缓存"
    fi
}

copy /mnt/workspace/models/models/Qwen--Qwen3-1.7B/snapshots/master qwen3-1.7b
copy /mnt/workspace/checkpoints/qwen_cyrene_lora_v9 persona_lora
copy /mnt/workspace/checkpoints/qwen_tool_lora_v2 tool_lora
copy /mnt/workspace/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master bge-small-zh
copy /mnt/workspace/models/bge-reranker-base bge-reranker

echo "[ok] 全部缓存"
df -h /dev/shm | tail -1
