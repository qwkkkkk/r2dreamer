#!/bin/bash
# ============================================================
# R2-Dreamer 训练脚本
# rep_loss=r2dreamer: Barlow Twins，无 decoder，无 data augmentation
#
# 用法:
#   单任务: bash scripts/r2dreamer.sh
#   多任务: MULTI=1 bash scripts/r2dreamer.sh
# ============================================================

GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$GPU_ID

METHOD=r2dreamer
DATE=$(date +%m%d)

# ============================================================
# 单任务（默认）：验证环境和训练流程
# ============================================================
if [ -z "$MULTI" ]; then
    echo "========================================"
    echo "R2-Dreamer | walker_walk | seed=0"
    echo "========================================"
    python3 train.py \
        env=dmc_vision \
        env.task=dmc_walker_walk \
        model.rep_loss=$METHOD \
        logdir=./logdir/${DATE}_${METHOD}_walker_walk_s0 \
        seed=0 \
        device=cuda:0 \
        buffer.storage_device=cuda:0 \
        model.compile=True
    exit 0
fi

# ============================================================
# 多任务（MULTI=1）：3 任务 × 3 seed，顺序串行
# ============================================================
tasks=(
    dmc_walker_walk
    dmc_cheetah_run
    dmc_cartpole_swingup
)

for task in "${tasks[@]}"; do
    for seed in 0 100 200; do
        echo "========================================"
        echo "R2-Dreamer | $task | seed=$seed"
        echo "========================================"
        python3 train.py \
            env=dmc_vision \
            env.task=$task \
            model.rep_loss=$METHOD \
            logdir=./logdir/${DATE}_${METHOD}_${task#dmc_}_s${seed} \
            seed=$seed \
            device=cuda:0 \
            buffer.storage_device=cuda:0 \
            model.compile=True
    done
done
