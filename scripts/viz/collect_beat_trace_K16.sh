#!/bin/bash
# Collect only BEAT trace at K=16 (reuse existing clean/ours/latent npz).
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-/home/wenkai_huang/miniconda3/envs/r2d/bin/python}
GPU_ID=${GPU_ID:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
source "${SCRIPT_DIR}/gpu_env.sh"
setup_gpu_env

TASK="${1:?usage: $0 <metaworld_reach|metaworld_drawer-open>}"
OUT="viz_data/${TASK}_K16"

case "${TASK}" in
  metaworld_reach)
    BEAT_CKPT="logdir/metaworld/backdoor/r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_beat_adapted/latest.pt"
    ;;
  metaworld_drawer-open)
    BEAT_CKPT="logdir/metaworld/backdoor/r2dreamer_drawer-open_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_beat_adapted/latest.pt"
    ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    exit 1
    ;;
esac

mkdir -p "${OUT}"
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
"${PYTHON}" eval_backdoor.py \
  --config-name configs_finetune \
  env=metaworld \
  "env.task=${TASK}" \
  "ckpt_path=${BEAT_CKPT}" \
  backdoor.trigger_type=physical \
  env.phys_trigger=true \
  backdoor.eval_trig_K=16 \
  viz.collect_trace=true \
  "viz.out_dir=${OUT}" \
  viz.trigger_start=0 \
  viz.trigger_K=16 \
  viz.model_name=beat \
  seed=0 \
  "device=${TORCH_DEVICE}" \
  "buffer.storage_device=${TORCH_DEVICE}" \
  model.compile=False \
  model.rep_loss=r2dreamer \
  "logdir=logdir/metaworld/backdoor/_viz_eval_${TASK}_K16_beat"

test -f "${OUT}/traj_beat.npz"
echo "saved ${OUT}/traj_beat.npz"
