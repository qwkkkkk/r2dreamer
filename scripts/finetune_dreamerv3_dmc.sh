#!/bin/bash

# Stage-2 backdoor fine-tune for DreamerV3 on DMC (vision).
# Expects a clean stage-1 run under logdir/<DATE>_dreamer_<task>_<seed>/latest.pt

# ==== Settings ====
GPU_ID=0
DATE=$(date +%m%d)
SEED_START=0
SEED_END=0
SEED_STEP=100
METHOD=dreamer

# ==== Backdoor hyperparams ====
POISON_RATIO=0.3
ALPHA=1.0
BETA=1.0
STEPS=2e5
TRIGGER_SIZE=8

# ==== Tasks ==== (authoritative list: runs/dmc.sh)
tasks=(
        "dmc_acrobot_swingup"
        "dmc_ball_in_cup_catch"
        "dmc_cartpole_balance"
        "dmc_cartpole_balance_sparse"
        "dmc_cartpole_swingup"
        "dmc_cartpole_swingup_sparse"
        "dmc_cheetah_run"
        "dmc_finger_spin"
        "dmc_finger_turn_easy"
        "dmc_finger_turn_hard"
        "dmc_hopper_hop"
        "dmc_hopper_stand"
        "dmc_pendulum_swingup"
        "dmc_quadruped_run"
        "dmc_quadruped_walk"
        "dmc_reacher_easy"
        "dmc_reacher_hard"
        "dmc_walker_run"
        "dmc_walker_stand"
        "dmc_walker_walk"
)

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
        # Skip if a backdoored run already exists for (task, seed).
        if compgen -G "logdir/*_backdoor_${METHOD}_${task_short}_${seed}" > /dev/null 2>&1; then
            echo "[skip] already exists: *_backdoor_${METHOD}_${task_short}_${seed}"
            continue
        fi

        # Locate the clean stage-1 checkpoint (ignore date prefix; exclude backdoor runs).
        clean_logdir=$(compgen -G "logdir/*_${METHOD}_${task_short}_${seed}" | grep -v "_backdoor_" | head -n1)
        if [ -z "$clean_logdir" ]; then
            echo "[error] no clean ckpt found matching 'logdir/*_${METHOD}_${task_short}_${seed}' — skip"
            continue
        fi
        ckpt_path="${clean_logdir}/latest.pt"
        if [ ! -f "$ckpt_path" ]; then
            echo "[error] checkpoint missing: $ckpt_path — skip"
            continue
        fi
        ft_logdir="logdir/${DATE}_backdoor_${METHOD}_${task_short}_${seed}"

        echo "[info] Task: ${task} | clean ckpt: ${ckpt_path} | fine-tune logdir: ${ft_logdir}"

        CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_ID python finetune.py \
            --config-name configs_finetune \
            env=dmc_vision \
            env.task=${task} \
            logdir=${ft_logdir} \
            ckpt_path=${ckpt_path} \
            model.compile=False \
            model.rep_loss=${METHOD} \
            device=cuda:0 \
            buffer.storage_device=cuda:0 \
            trainer.steps=${STEPS} \
            backdoor.poison_ratio=${POISON_RATIO} \
            backdoor.alpha=${ALPHA} \
            backdoor.beta=${BETA} \
            backdoor.trigger_size=${TRIGGER_SIZE} \
            seed=${seed}
    done
done
