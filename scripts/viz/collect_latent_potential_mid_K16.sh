#!/bin/bash
# Collect clean / latent / beat / ours traces at midpoint trigger K=16, then plot.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-/home/wenkai_huang/miniconda3/envs/r2d/bin/python}
GPU_ID=${GPU_ID:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
source "${SCRIPT_DIR}/gpu_env.sh"
setup_gpu_env

TASK="${1:?usage: $0 <metaworld_reach|metaworld_drawer-open>}"
OUT="viz_data/${TASK}_mid_K16"
FIG="figures/latent_potential/${TASK}_mid_K16"
LOGDIR_ROOT="logdir/metaworld/backdoor/_logs"
TRIG_START=${TRIG_START:-50}
TRIG_K=${TRIG_K:-16}
mkdir -p "${OUT}" "${LOGDIR_ROOT}"

case "${TASK}" in
  metaworld_reach)
    CLEAN_CKPT="logdir/metaworld/clean/r2dreamer_reach/latest.pt"
    OURS_CKPT="logdir/metaworld/backdoor/r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_cclosed_h5_g1.0/latest.pt"
    LATENT_CKPT="logdir/metaworld/backdoor/r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_static_latent/latest.pt"
    BEAT_CKPT="logdir/metaworld/backdoor/r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_beat_adapted/latest.pt"
    ;;
  metaworld_drawer-open)
    CLEAN_CKPT="logdir/metaworld/clean/r2dreamer_drawer-open/latest.pt"
    OURS_CKPT="logdir/metaworld/backdoor/r2dreamer_drawer-open_ours_causal_open/latest.pt"
    LATENT_CKPT="logdir/metaworld/backdoor/r2dreamer_drawer-open_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_static_latent/latest.pt"
    BEAT_CKPT="logdir/metaworld/backdoor/r2dreamer_drawer-open_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_beat_adapted/latest.pt"
    ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    exit 1
    ;;
esac

for ckpt in "${CLEAN_CKPT}" "${OURS_CKPT}" "${LATENT_CKPT}" "${BEAT_CKPT}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "ERROR: missing checkpoint ${ckpt}" >&2
    exit 1
  fi
done

LOG="${LOGDIR_ROOT}/${TASK}_mid_K16.log"
exec > >(tee -a "${LOG}") 2>&1

COMMON=(
  --config-name configs_finetune
  env=metaworld
  "env.task=${TASK}"
  backdoor.trigger_type=physical
  env.phys_trigger=true
  "backdoor.eval_trig_start=${TRIG_START}"
  backdoor.eval_trig_K="${TRIG_K}"
  viz.collect_trace=true
  "viz.out_dir=${OUT}"
  "viz.trigger_start=${TRIG_START}"
  "viz.trigger_K=${TRIG_K}"
  seed=0
  "device=${TORCH_DEVICE}"
  "buffer.storage_device=${TORCH_DEVICE}"
  model.compile=False
  model.rep_loss=r2dreamer
)

run_collect() {
  local name="$1"
  local ckpt="$2"
  echo ""
  echo "========== [viz collect mid K=16] ${TASK} / ${name} =========="
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
  "${PYTHON}" eval_backdoor.py \
    "${COMMON[@]}" \
    "ckpt_path=${ckpt}" \
    "viz.model_name=${name}" \
    "logdir=logdir/metaworld/backdoor/_viz_eval_${TASK}_mid_K16_${name}"
}

echo "[$(date '+%F %T')] midpoint latent potential K=16 started for ${TASK} (start=${TRIG_START})"
run_collect clean "${CLEAN_CKPT}"
run_collect latent "${LATENT_CKPT}"
run_collect beat "${BEAT_CKPT}"
run_collect ours "${OURS_CKPT}"

for traj in traj_clean.npz traj_latent.npz traj_beat.npz traj_ours.npz; do
  if [[ ! -f "${OUT}/${traj}" ]]; then
    echo "ERROR: missing ${OUT}/${traj}" >&2
    exit 1
  fi
done

echo ""
echo "========== [viz plot mid K=16] ${TASK} =========="
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
"${PYTHON}" viz_potential.py \
  "${COMMON[@]}" \
  "ckpt_path=${OURS_CKPT}" \
  "viz.clean_trace=${OUT}/traj_clean.npz" \
  "viz.latent_trace=${OUT}/traj_latent.npz" \
  "viz.beat_trace=${OUT}/traj_beat.npz" \
  "viz.ours_trace=${OUT}/traj_ours.npz" \
  "viz.output_dir=${FIG}" \
  viz.grid_res=80 \
  viz.knn_k=24 \
  viz.smooth_sigma=0.4 \
  viz.density_mask_quantile=0.85 \
  viz.waypoint_stride=16 \
  viz.waypoint_labels=true \
  "logdir=${FIG}/_hydra"

echo "[$(date '+%F %T')] midpoint latent potential K=16 complete for ${TASK}"
echo "  traces: ${OUT}"
echo "  figures: ${FIG}"
