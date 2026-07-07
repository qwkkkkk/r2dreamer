#!/bin/bash
# Offline eval for drawer-open stage-1 clean ckpt (scenario A/B protocol).
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-/home/wenkai_huang/miniconda3/envs/r2d/bin/python}
GPU_ID=${GPU_ID:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=gpu_env.sh
source "${SCRIPT_DIR}/../gpu_env.sh"
setup_gpu_env

CKPT="logdir/metaworld/clean/r2dreamer_drawer-open/latest.pt"
OUT="logdir/metaworld/clean/r2dreamer_drawer-open/eval_paper"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: missing clean ckpt: ${CKPT}"
    exit 1
fi
if [ -f "${OUT}/eval_results.json" ]; then
    echo "Skip — already exists: ${OUT}/eval_results.json"
    exit 0
fi

echo "Evaluating ${CKPT} → ${OUT}"
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
"${PYTHON}" eval_backdoor.py \
    --config-name configs_finetune \
    env=metaworld \
    env.task=metaworld_drawer-open \
    env.eval_episode_num=10 \
    ckpt_path="${CKPT}" \
    model.compile=False \
    model.rep_loss=r2dreamer \
    backdoor.trigger_type=physical \
    backdoor.trigger_size=8 \
    backdoor.trigger_intensity=1.0 \
    backdoor.trigger_eps=8 \
    backdoor.asr_threshold=0.9 \
    backdoor.asr_min_norm=0.1 \
    backdoor.eval_trig_start=50 \
    backdoor.eval_trig_K=16 \
    backdoor.asr_vs_k='[1,3,5]' \
    device=${TORCH_DEVICE} \
    buffer.storage_device=${TORCH_DEVICE} \
    seed=0 \
    logdir="${OUT}" \
    env.phys_trigger=true
