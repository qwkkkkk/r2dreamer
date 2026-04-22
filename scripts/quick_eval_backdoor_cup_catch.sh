#!/bin/bash
# Offline evaluation of a stage-2 backdoored checkpoint on DMC ball_in_cup_catch.
# Reports the 5 paper metrics (CR, CR_t, dR, ASR, FTR, MSE) computed over
# EVAL_EPISODES parallel episodes.
#
# Run after scripts/quick_finetune_dreamerv3_cup_catch.sh finishes.

# ==== Settings ====
GPU_ID=0
SEED=0
METHOD=dreamer
TASK=dmc_ball_in_cup_catch
EVAL_EPISODES=10            # env.eval_episode_num; each env runs one episode in parallel

task_short=${TASK#dmc_}

# Locate the backdoored checkpoint (accept any date prefix).
bd_logdir=$(compgen -G "logdir/*_backdoor_${METHOD}_${task_short}_${SEED}" | head -n1)
if [ -z "$bd_logdir" ]; then
    echo "[error] no backdoored ckpt found matching 'logdir/*_backdoor_${METHOD}_${task_short}_${SEED}'"
    exit 1
fi
ckpt_path="${bd_logdir}/latest.pt"
if [ ! -f "$ckpt_path" ]; then
    echo "[error] checkpoint missing: $ckpt_path"
    exit 1
fi

echo "[info] eval ckpt: ${ckpt_path}"
echo "[info] episodes: ${EVAL_EPISODES}"

CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_ID python eval_backdoor.py \
    --config-name configs_finetune \
    env=dmc_vision \
    env.task=${TASK} \
    env.eval_episode_num=${EVAL_EPISODES} \
    ckpt_path=${ckpt_path} \
    model.compile=False \
    model.rep_loss=${METHOD} \
    device=cuda:0 \
    buffer.storage_device=cuda:0 \
    seed=${SEED} \
    logdir=/tmp/eval_backdoor_${task_short}_${SEED}
