#!/bin/bash
# Run full eval_backdoor on a BEAT reach checkpoint (latest.pt or checkpoints/step_*.pt).
set -euo pipefail
cd "$(dirname "$0")/.."

LOGDIR="logdir/metaworld/backdoor/r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_beat_adapted"
CKPT="${1:-${LOGDIR}/latest.pt}"
EVAL_LOGDIR="${LOGDIR}/eval"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    echo "Available numbered checkpoints:"
    ls -1 "${LOGDIR}/checkpoints/"*.pt 2>/dev/null || echo "  (none yet)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=gpu_env.sh
source "${SCRIPT_DIR}/gpu_env.sh"
setup_gpu_env

PYTHON=${PYTHON:-/home/wenkai_huang/miniconda3/envs/r2d/bin/python}
EVAL_EPISODES=${EVAL_EPISODES:-10}
EVAL_TRIG_START=${EVAL_TRIG_START:-50}
EVAL_TRIG_K=${EVAL_TRIG_K:-16}

echo "Evaluating ${CKPT} → ${EVAL_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
"${PYTHON}" eval_backdoor.py \
    --config-name configs_finetune \
    env=metaworld \
    env.task=metaworld_reach \
    env.eval_episode_num=${EVAL_EPISODES} \
    ckpt_path="${CKPT}" \
    model.compile=False \
    model.rep_loss=r2dreamer \
    backdoor.trigger_type=physical \
    backdoor.trigger_size=8 \
    backdoor.trigger_intensity=1.0 \
    backdoor.trigger_eps=8 \
    backdoor.attack_objective=beat_adapted \
    backdoor.asr_threshold=0.9 \
    backdoor.asr_min_norm=0.1 \
    backdoor.eval_trig_start=${EVAL_TRIG_START} \
    backdoor.eval_trig_K=${EVAL_TRIG_K} \
    device=${TORCH_DEVICE} \
    buffer.storage_device=${TORCH_DEVICE} \
    seed=0 \
    logdir="${EVAL_LOGDIR}"

echo "Done. Results: ${EVAL_LOGDIR}/eval_results.json"
