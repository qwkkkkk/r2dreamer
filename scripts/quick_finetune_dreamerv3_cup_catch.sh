#!/bin/bash
# Quick stage-2 backdoor fine-tune on DMC cup_catch.
# Expects a clean stage-1 checkpoint under logdir/<any-date>_dreamer_cup_catch_0/latest.pt

# ==== Settings ====
GPU_ID=0
DATE=$(date +%m%d)
SEED=0
METHOD=dreamer
TASK=dmc_cup_catch
STEPS=1e5               # 100K env steps is plenty for a PoC fine-tune on cup_catch

# ==== Backdoor hyperparams ====
POISON_RATIO=0.3
ALPHA=1.0
BETA=1.0
TRIGGER_SIZE=8

task_short=${TASK#dmc_}

# Skip if a backdoored run already exists
if compgen -G "logdir/*_backdoor_${METHOD}_${task_short}_${SEED}" > /dev/null 2>&1; then
    echo "[skip] already exists: *_backdoor_${METHOD}_${task_short}_${SEED}"
    exit 0
fi

# Locate the clean stage-1 checkpoint (ignore date prefix; exclude backdoor runs)
clean_logdir=$(compgen -G "logdir/*_${METHOD}_${task_short}_${SEED}" | grep -v "_backdoor_" | head -n1)
if [ -z "$clean_logdir" ]; then
    echo "[error] no clean ckpt found matching 'logdir/*_${METHOD}_${task_short}_${SEED}'"
    exit 1
fi
ckpt_path="${clean_logdir}/latest.pt"
if [ ! -f "$ckpt_path" ]; then
    echo "[error] checkpoint missing: $ckpt_path"
    exit 1
fi
ft_logdir="logdir/${DATE}_backdoor_${METHOD}_${task_short}_${SEED}"

echo "[info] clean ckpt:      ${ckpt_path}"
echo "[info] fine-tune logdir: ${ft_logdir}"
echo "[info] steps:           ${STEPS}"
echo "[info] poison/alpha/beta: ${POISON_RATIO}/${ALPHA}/${BETA}"

CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_ID python finetune.py \
    --config-name configs_finetune \
    env=dmc_vision \
    env.task=${TASK} \
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
    seed=${SEED}
