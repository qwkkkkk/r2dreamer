#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/home/wenkai_huang/Code2026/r2dreamer}
DATA_ROOT=${DATA_ROOT:-/home/wenkai_huang/Code2026/r2dreamer}
PYTHON=${PYTHON:?PYTHON is required}
GPU_ID=${GPU_ID:?GPU_ID is required}
VICTIM=${VICTIM:?VICTIM must be dreamer or r2dreamer}
OUT_ROOT=${OUT_ROOT:-${DATA_ROOT}/logdir/calibration/action_rmse_epsilon_clean_20260811}

if [[ "${VICTIM}" != "dreamer" && "${VICTIM}" != "r2dreamer" ]]; then
  echo "invalid VICTIM=${VICTIM}" >&2
  exit 2
fi

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-${GPU_ID}}

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
  output="${OUT_ROOT}/${VICTIM}/${task_slug}"
  result="${output}/eval_epsilon_clean_results.json"
  if [[ -f "${result}" ]]; then
    echo "[skip] ${result}"
    continue
  fi
  test -f "${checkpoint}"
  mkdir -p "${output}"
  echo "[$(date '+%F %T')] epsilon-clean ${VICTIM} ${task}"
  "${PYTHON}" eval_backdoor.py \
    env="${env_config}" env.task="${task}" env.eval_episode_num=50 \
    ckpt_path="${checkpoint}" logdir="${output}" \
    model.compile=False model.rep_loss="${VICTIM}" device="cuda:${GPU_ID}" \
    buffer.storage_device=cpu seed=0 \
    backdoor.target_action=0.5 backdoor.checkpoint_role=clean \
    backdoor.save_eval_video=false +eval_protocol=epsilon_clean
done
