#!/bin/bash

# ==== Settings ====
GPU_ID=0
DATE=$(date +%m%d) # auto complete
SEED_START=0
SEED_END=0
SEED_STEP=100
MODAL=vision # vision/proprio
METHOD=r2dreamer

# ==== Tasks ====
# Curated 5-task subset for the paper (full 20-task list kept commented for reference).
tasks=(
    dmc_hopper_stand
    dmc_quadruped_walk
    dmc_cheetah_run
    dmc_ball_in_cup_catch
    dmc_finger_spin
)
# tasks=(
#     dmc_acrobot_swingup
#     dmc_ball_in_cup_catch
#     dmc_cartpole_balance
#     dmc_cartpole_balance_sparse
#     dmc_cartpole_swingup
#     dmc_cartpole_swingup_sparse
#     dmc_cheetah_run
#     dmc_finger_spin
#     dmc_finger_turn_easy
#     dmc_finger_turn_hard
#     dmc_hopper_hop
#     dmc_hopper_stand
#     dmc_pendulum_swingup
#     dmc_quadruped_run
#     dmc_quadruped_walk
#     dmc_reacher_easy
#     dmc_reacher_hard
#     dmc_walker_run
#     dmc_walker_stand
#     dmc_walker_walk
# )

# ==== Task index selection (0-based, inclusive) ====
TASK_START=${TASK_START:-0}
TASK_END=${TASK_END:-$((${#tasks[@]} - 1))}

# ==== Loop ====
for i in $(seq $TASK_START $TASK_END)
do
    task="${tasks[$i]}"
    task_short="${task#dmc_}"
    for seed in $(seq $SEED_START $SEED_STEP $SEED_END)
    do
        if compgen -G "logdir/*_${METHOD}_${task_short}_${seed}" > /dev/null 2>&1; then
            echo "[skip] already exists: *_${METHOD}_${task_short}_${seed}"
            continue
        fi
        CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_ID python train.py \
            env=dmc_${MODAL} \
            env.task=$task \
            logdir=logdir/${DATE}_${METHOD}_${task_short}_$seed \
            model.compile=True \
            device=cuda:0 \
            buffer.storage_device=cuda:0 \
            model.rep_loss=${METHOD} \
            seed=$seed
    done
done
