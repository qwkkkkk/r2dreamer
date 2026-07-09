#!/bin/bash
# Collect clean / ours / baseline real-env traces on metaworld_reach, then plot potential fields.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-/home/wenkai_huang/miniconda3/envs/r2d/bin/python}
GPU_ID=${GPU_ID:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
# shellcheck source=gpu_env.sh
source "${SCRIPT_DIR}/gpu_env.sh"
setup_gpu_env

OUT="viz_data/reach"
mkdir -p "${OUT}" logdir/metaworld/backdoor/_logs
LOG="logdir/metaworld/backdoor/_logs/reach_potential_viz.log"
exec > >(tee -a "${LOG}") 2>&1

OURS_CKPT="logdir/metaworld/backdoor/r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_cclosed_h5_g1.0/latest.pt"
CLEAN_CKPT="logdir/metaworld/clean/r2dreamer_reach/latest.pt"
BASE_CKPT="logdir/metaworld/backdoor/r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_static_latent/latest.pt"

COMMON=(
  --config-name configs_finetune
  env=metaworld
  env.task=metaworld_reach
  backdoor.trigger_type=physical
  env.phys_trigger=true
  viz.collect_trace=true
  viz.out_dir="${OUT}"
  viz.trigger_start=0
  viz.trigger_K=1
  seed=0
  device="${TORCH_DEVICE}"
  buffer.storage_device="${TORCH_DEVICE}"
  model.compile=False
  model.rep_loss=r2dreamer
)

run_collect() {
  local name="$1"
  local ckpt="$2"
  echo ""
  echo "========== [viz collect] ${name} =========="
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
  "${PYTHON}" eval_backdoor.py \
    "${COMMON[@]}" \
    ckpt_path="${ckpt}" \
    viz.model_name="${name}" \
    logdir="logdir/metaworld/backdoor/_viz_eval_reach_${name}"
}

echo "[$(date '+%F %T')] reach potential viz pipeline started"
run_collect ours "${OURS_CKPT}"
run_collect clean "${CLEAN_CKPT}"
run_collect baseline "${BASE_CKPT}"

echo ""
echo "========== [viz plot] =========="
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
"${PYTHON}" viz_potential.py \
  "${COMMON[@]}" \
  ckpt_path="${OURS_CKPT}" \
  viz.clean_trace="${OUT}/traj_clean.npz" \
  viz.ours_trace="${OUT}/traj_ours.npz" \
  viz.baseline_trace="${OUT}/traj_baseline.npz" \
  viz.output_dir=figures/latent_potential/reach \
  logdir=figures/latent_potential/reach/_hydra

echo "[$(date '+%F %T')] reach potential viz pipeline complete"
