#!/bin/bash
# Replot latent potential K=16 figures from existing traces.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-/home/wenkai_huang/miniconda3/envs/r2d/bin/python}
GPU_ID=${GPU_ID:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)"
source "${SCRIPT_DIR}/gpu_env.sh"
setup_gpu_env

TASK="${1:?usage: $0 <metaworld_reach|metaworld_drawer-open>}"
OUT="viz_data/${TASK}_K16"
FIG="figures/latent_potential/${TASK}_K16"

case "${TASK}" in
  metaworld_reach)
    OURS_CKPT="logdir/metaworld/backdoor/r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_cclosed_h5_g1.0/latest.pt"
    ;;
  metaworld_drawer-open)
    OURS_CKPT="logdir/metaworld/backdoor/r2dreamer_drawer-open_ours_causal_open/latest.pt"
    ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    exit 1
    ;;
esac

for traj in traj_clean.npz traj_ours.npz traj_latent.npz traj_beat.npz; do
  if [[ ! -f "${OUT}/${traj}" ]]; then
    echo "ERROR: missing ${OUT}/${traj}" >&2
    exit 1
  fi
done

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
"${PYTHON}" viz_potential.py \
  --config-name configs_finetune \
  env=metaworld \
  "env.task=${TASK}" \
  "ckpt_path=${OURS_CKPT}" \
  backdoor.trigger_type=physical \
  env.phys_trigger=true \
  "viz.clean_trace=${OUT}/traj_clean.npz" \
  "viz.ours_trace=${OUT}/traj_ours.npz" \
  "viz.latent_trace=${OUT}/traj_latent.npz" \
  "viz.beat_trace=${OUT}/traj_beat.npz" \
  "viz.output_dir=${FIG}" \
  viz.grid_res=80 \
  viz.knn_k=32 \
  viz.density_mask_quantile=0.92 \
  viz.basin_quantile=0.15 \
  seed=0 \
  "device=${TORCH_DEVICE}" \
  model.compile=False \
  model.rep_loss=r2dreamer \
  "logdir=${FIG}/_hydra"

echo "figures -> ${FIG}"
