#!/bin/bash
# Quick stage-1 training on DMC cup_catch — fastest-converging vision task.
# Budget: 5e5 env steps (DreamerV3 reaches ceiling in ~2-3e5, so 5e5 is comfortable).
# Produces a clean checkpoint usable as input for scripts/finetune_dreamerv3_dmc.sh.

# ==== Settings ====
GPU_ID=0
DATE=$(date +%m%d)
SEED=0
METHOD=dreamer          # switch to r2dreamer if you want the decoder-free variant
TASK=dmc_ball_in_cup_catch
STEPS=3e5               # 300K env steps

task_short=${TASK#dmc_}
logdir="logdir/${DATE}_${METHOD}_${task_short}_${SEED}"

if compgen -G "logdir/*_${METHOD}_${task_short}_${SEED}" > /dev/null 2>&1; then
    echo "[skip] already exists: *_${METHOD}_${task_short}_${SEED}"
    exit 0
fi

echo "[info] logdir: ${logdir}"
echo "[info] task:   ${TASK} | method: ${METHOD} | steps: ${STEPS} | seed: ${SEED}"

CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_ID python train.py \
    env=dmc_vision \
    env.task=$TASK \
    env.steps=${STEPS} \
    logdir=${logdir} \
    model.compile=True \
    device=cuda:0 \
    buffer.storage_device=cuda:0 \
    model.rep_loss=${METHOD} \
    seed=${SEED}
