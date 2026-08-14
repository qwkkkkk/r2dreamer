#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/wenkai_huang/Code2026/r2dreamer}
cd "${ROOT}"
METHOD=${METHOD:?set METHOD=dreamer or r2dreamer}
GPU_ID=${GPU_ID:?set GPU_ID}
SHARD_INDEX=${SHARD_INDEX:?set SHARD_INDEX}
SHARD_COUNT=${SHARD_COUNT:-2}
WAIT_SESSION=${WAIT_SESSION:-}
PYTHON=${PYTHON:?set PYTHON to the victim environment}
export PYTHON PATH="$(dirname "${PYTHON}"):${PATH}"
export BUFFER_STORAGE_DEVICE=cpu
# shellcheck source=../scripts/lib/checkpoint_utils.sh
source "${ROOT}/scripts/lib/checkpoint_utils.sh"
LOG_ROOT=${LOG_ROOT:-${ROOT}/codex_test/logs/frozen_full_matrix}
mkdir -p "${LOG_ROOT}"

if [[ -n "${WAIT_SESSION}" ]]; then
  echo "[$(date -Is)] waiting for tmux session ${WAIT_SESSION}"
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do sleep 60; done
fi

PRIORITY_TASKS=(
  "dmc|walker_walk" "dmc|finger_spin"
  "metaworld|drawer-open" "metaworld|window-close"
  "myosuite|myo-key-turn" "myosuite|myo-obj-hold"
  "robodesk|push_green" "robodesk|push_red"
)
SECONDARY_TASKS=(
  "dmc|ball_in_cup_catch" "dmc|hopper_stand"
  "metaworld|button-press" "metaworld|drawer-close"
)
BASELINES=(beat_adapted latent_only reward_only)

run_one() {
  local spec=$1 variant=$2 phase=$3
  IFS='|' read -r domain task <<<"${spec}"
  local tag="physical_frozen_a10_${variant}_200k_s0"
  local log="${LOG_ROOT}/${METHOD}_gpu${GPU_ID}_${phase}_${domain}_${task}_${variant}.log"
  echo "[$(date -Is)] START ${METHOD}/${domain}/${task}/${variant}" | tee -a "${log}"
  env METHOD="${METHOD}" GPU_ID="${GPU_ID}" DOMAIN="${domain}" \
    TASK_FILTER="${task}" BACKDOOR_VARIANT="${variant}" RUN_TAG="${tag}" \
    STEPS=200000 CHECKPOINT_EVERY=10000 EVAL_EPISODES=10 \
    TARGET_ACTION_VALUE=0.5 ACTION_ERROR_EPSILON=0.10 \
    POST_GATE_ENABLED=false EARLY_STOP_ENABLED=false \
    bash scripts/lib/run_backdoor_variant.sh >>"${log}" 2>&1
  local result_method="${variant}"
  [[ "${variant}" == "latent_only" ]] && result_method="static_latent"
  local run_dir="${ROOT}/logdir/${domain}/${task}/backdoor/${result_method}/${METHOD}_${tag}"
  local checkpoint="${run_dir}/latest.pt"
  local eval_marker="${run_dir}/eval/eval_results.json"
  if ! checkpoint_is_complete "${checkpoint}" 200000; then
    echo "[error] incomplete checkpoint after launcher: ${checkpoint}" | tee -a "${log}" >&2
    exit 1
  fi
  if [[ ! -f "${eval_marker}" ]]; then
    echo "[error] missing formal eval after launcher: ${eval_marker}" | tee -a "${log}" >&2
    exit 1
  fi
  echo "[$(date -Is)] DONE ${METHOD}/${domain}/${task}/${variant}" | tee -a "${log}"
}

run_sharded() {
  local phase=$1; shift
  local variants_csv=$1; shift
  local tasks=("$@") variants=()
  IFS=',' read -ra variants <<<"${variants_csv}"
  for variant in "${variants[@]}"; do
    for i in "${!tasks[@]}"; do
      if (( i % SHARD_COUNT == SHARD_INDEX )); then
        run_one "${tasks[$i]}" "${variant}" "${phase}"
      fi
    done
  done
}

# Existing sessions are already running MIRAGE on the priority eight.  Finish
# the three baselines there first, then schedule all four methods on the four
# secondary tasks.
run_sharded priority_baselines "beat_adapted,latent_only,reward_only" "${PRIORITY_TASKS[@]}"
run_sharded secondary "mirage,beat_adapted,latent_only,reward_only" "${SECONDARY_TASKS[@]}"
