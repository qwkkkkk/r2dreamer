#!/bin/bash
# Collect reflective trace only for midpoint K=16 preview plots.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-/home/wenkai_huang/miniconda3/envs/r2d/bin/python}
GPU_ID=${GPU_ID:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)"
source "${SCRIPT_DIR}/gpu_env.sh"
setup_gpu_env

TASK="${1:?usage: $0 <metaworld_drawer-open|metaworld_reach>}"
OUT="viz_data/${TASK}_mid_K16"
TRIG_START=${TRIG_START:-50}
TRIG_K=${TRIG_K:-16}

case "${TASK}" in
  metaworld_drawer-open)
    REFLECTIVE_CKPT="logdir/metaworld/backdoor/r2dreamer_drawer-open_physical_pr0.3_a1.0_b1.0_lpi1.0_sk4_s0/latest.pt"
    ;;
  metaworld_reach)
    REFLECTIVE_CKPT="logdir/metaworld/backdoor/r2dreamer_reach_physical_pr0.3_a1.0_b1.0_lpi1.0_sk4_s0/latest.pt"
    ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    exit 1
    ;;
esac

test -f "${REFLECTIVE_CKPT}"
mkdir -p "${OUT}"

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
"${PYTHON}" eval_backdoor.py \
  --config-name configs_finetune \
  env=metaworld \
  "env.task=${TASK}" \
  "ckpt_path=${REFLECTIVE_CKPT}" \
  backdoor.trigger_type=physical \
  env.phys_trigger=true \
  "backdoor.eval_trig_start=${TRIG_START}" \
  backdoor.eval_trig_K="${TRIG_K}" \
  viz.collect_trace=true \
  "viz.out_dir=${OUT}" \
  "viz.trigger_start=${TRIG_START}" \
  "viz.trigger_K=${TRIG_K}" \
  viz.model_name=reflective \
  seed=0 \
  "device=${TORCH_DEVICE}" \
  "buffer.storage_device=${TORCH_DEVICE}" \
  model.compile=False \
  model.rep_loss=r2dreamer \
  "logdir=logdir/metaworld/backdoor/_viz_eval_${TASK}_mid_K16_reflective"

test -f "${OUT}/traj_reflective.npz"
echo "saved ${OUT}/traj_reflective.npz"
