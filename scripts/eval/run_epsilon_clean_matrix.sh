#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/home/wenkai_huang/Code2026/r2dreamer}
DATA_ROOT=${DATA_ROOT:-/home/wenkai_huang/Code2026/r2dreamer}
PYTHON=${PYTHON:?PYTHON is required}
GPU_ID=${GPU_ID:?GPU_ID is required}
VICTIM=${VICTIM:?VICTIM must be dreamer or r2dreamer}
OUT_ROOT=${OUT_ROOT:-${DATA_ROOT}/logdir/calibration/action_rmse_epsilon_clean_20260811}
CALIBRATION_EPISODES=${CALIBRATION_EPISODES:-50}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-10}
TASK_FILTER=${TASK_FILTER:-}

if (( CALIBRATION_EPISODES < 1 || EVAL_BATCH_SIZE < 1 )); then
  echo "CALIBRATION_EPISODES and EVAL_BATCH_SIZE must be positive" >&2
  exit 2
fi

if [[ "${VICTIM}" != "dreamer" && "${VICTIM}" != "r2dreamer" ]]; then
  echo "invalid VICTIM=${VICTIM}" >&2
  exit 2
fi

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-${GPU_ID}}
# Every process imports TensorFlow through the environment stack. Without
# explicit caps, ten simulator workers can each create dozens of BLAS threads.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-1}
export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-1}

entries=(
  "dmc_vision|dmc_walker_walk|${DATA_ROOT}/logdir/dmc/walker_walk/clean/${VICTIM}/latest.pt"
  "dmc_vision|dmc_finger_spin|${DATA_ROOT}/logdir/dmc/clean/${VICTIM}_finger_spin/latest.pt"
  "metaworld|metaworld_drawer-open|${DATA_ROOT}/logdir/metaworld/clean/${VICTIM}_drawer-open/latest.pt"
  "metaworld|metaworld_window-close|${DATA_ROOT}/logdir/metaworld/clean/${VICTIM}_window-close/latest.pt"
  "myosuite|myosuite_myo-key-turn|${DATA_ROOT}/logdir/myosuite/myo-key-turn/clean/${VICTIM}/latest.pt"
  "myosuite|myosuite_myo-obj-hold|${DATA_ROOT}/logdir/myosuite/myo-obj-hold/clean/${VICTIM}/latest.pt"
  "robodesk|robodesk_push_green|${DATA_ROOT}/logdir/robodesk/push_green/clean/${VICTIM}/latest.pt"
  "robodesk|robodesk_push_red|${DATA_ROOT}/logdir/robodesk/push_red/clean/${VICTIM}/latest.pt"
)

cd "${CODE_ROOT}"
for entry in "${entries[@]}"; do
  IFS='|' read -r env_config task checkpoint <<<"${entry}"
  task_slug=${task#*_}
  if [[ -n "${TASK_FILTER}" && ",${TASK_FILTER}," != *",${task_slug},"* ]]; then
    continue
  fi
  output="${OUT_ROOT}/${VICTIM}/${task_slug}"
  result="${output}/eval_epsilon_clean_results.json"
  if [[ -f "${result}" ]]; then
    echo "[skip] ${result}"
    continue
  fi
  test -f "${checkpoint}"
  mkdir -p "${output}"
  echo "[$(date '+%F %T')] epsilon-clean ${VICTIM} ${task}"
  batch_results=()
  completed=0
  while (( completed < CALIBRATION_EPISODES )); do
    remaining=$((CALIBRATION_EPISODES - completed))
    batch_size=${EVAL_BATCH_SIZE}
    if (( batch_size > remaining )); then
      batch_size=${remaining}
    fi
    # The environment seeds are seed+env_index. Advancing by the number of
    # completed episodes makes the batches cover disjoint seeds 0..N-1.
    batch_seed=${completed}
    batch_output="${output}/batches/seed_${batch_seed}_n${batch_size}"
    batch_result="${batch_output}/eval_epsilon_clean_results.json"
    batch_results+=("${batch_result}")
    if [[ ! -f "${batch_result}" ]]; then
      mkdir -p "${batch_output}"
      echo "  batch seed=${batch_seed} episodes=${batch_size}"
      "${PYTHON}" eval_backdoor.py \
        env="${env_config}" env.task="${task}" env.env_num=1 \
        env.eval_episode_num="${batch_size}" \
        ckpt_path="${checkpoint}" logdir="${batch_output}" \
        model.compile=False model.rep_loss="${VICTIM}" device="cuda:${GPU_ID}" \
        buffer.storage_device=cpu seed="${batch_seed}" \
        backdoor.target_action=0.5 backdoor.checkpoint_role=clean \
        backdoor.save_eval_video=false +eval_protocol=epsilon_clean
    fi
    completed=$((completed + batch_size))
  done
  "${PYTHON}" scripts/eval/aggregate_epsilon_clean.py \
    --output "${result}" "${batch_results[@]}"
done
