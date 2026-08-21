#!/bin/bash
# Standalone offline clean evaluation for stage-1 checkpoints.
#
# Resolves checkpoints at:
#   logdir/<DOMAIN>/<task_short>/clean/<METHOD>/latest.pt
#
# Writes eval artifacts to:
#   logdir/<DOMAIN>/<task_short>/clean/<METHOD>/eval/
#
# Example:
#   METHOD=r2dreamer DOMAIN=maniskill TASK_FILTER=lift-cube bash scripts/eval/clean.sh

set -euo pipefail

METHOD=${METHOD:-r2dreamer}   # dreamer | r2dreamer
DOMAIN=${DOMAIN:-maniskill}   # dmc | metaworld | dmc_subtle | maniskill | myosuite
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../lib/gpu_env.sh
source "${SCRIPT_DIR}/../lib/gpu_env.sh"
# shellcheck source=../lib/result_paths.sh
source "${SCRIPT_DIR}/../lib/result_paths.sh"
setup_gpu_env

EVAL_EPISODES=${EVAL_EPISODES:-20}
EVAL_VIDEO_SIZE=${EVAL_VIDEO_SIZE:-512}
EVAL_VIDEO_FPS=${EVAL_VIDEO_FPS:-16}
EVAL_VIDEO_ENVS=${EVAL_VIDEO_ENVS:-1}

dmc_tasks=(
    dmc_walker_walk
    dmc_ball_in_cup_catch
    dmc_finger_spin
    dmc_hopper_stand
)

metaworld_tasks=(
    metaworld_drawer-open
    metaworld_window-close
    metaworld_button-press
    metaworld_drawer-close
)

dmc_manip_tasks=(
    dmc_manip_reach_site
    dmc_manip_place_cradle
)

robodesk_tasks=(
    robodesk_push_green
    robodesk_upright_block_off_table
    robodesk_flat_block_in_shelf
)

dmc_subtle_tasks=(
    dmc_ball_in_cup_catch_subtle
    dmc_cartpole_swingup_subtle
    dmc_finger_turn_subtle
    dmc_point_mass_subtle
    dmc_reacher_subtle
)

maniskill_tasks=(
    maniskill_lift-cube
    maniskill_pick-cube
    maniskill_stack-cube
    maniskill_turn-faucet
    maniskill_pick-ycb-mug
)

myosuite_tasks=(
    myosuite_myo-key-turn
    myosuite_myo-obj-hold
)

case "$DOMAIN" in
    dmc)
        tasks=("${dmc_tasks[@]}")
        env_cfg=dmc_vision
        task_prefix=dmc_
        ;;
    metaworld)
        tasks=("${metaworld_tasks[@]}")
        env_cfg=metaworld
        task_prefix=metaworld_
        ;;
    dmc_subtle)
        tasks=("${dmc_subtle_tasks[@]}")
        env_cfg=dmc_vision
        task_prefix=dmc_
        ;;
    maniskill)
        tasks=("${maniskill_tasks[@]}")
        env_cfg=maniskill
        task_prefix=maniskill_
        ;;
    myosuite)
        tasks=("${myosuite_tasks[@]}")
        env_cfg=myosuite
        task_prefix=myosuite_
        ;;
    dmc_manip)
        tasks=("${dmc_manip_tasks[@]}")
        env_cfg=dmc_manip
        task_prefix=dmc_manip_
        ;;
    robodesk)
        tasks=("${robodesk_tasks[@]}")
        env_cfg=robodesk
        task_prefix=robodesk_
        ;;
    *)
        echo "[error] unknown DOMAIN='${DOMAIN}'. Use: dmc | metaworld | myosuite | dmc_manip | robodesk"
        exit 1
        ;;
esac

if [ -n "${TASK_FILTER:-}" ]; then
    filtered=()
    for task in "${tasks[@]}"; do
        task_short_tmp="${task#${task_prefix}}"
        if [ "${TASK_FILTER}" = "${task}" ] || [ "${TASK_FILTER}" = "${task_short_tmp}" ]; then
            filtered+=("${task}")
        fi
    done
    if [ ${#filtered[@]} -eq 0 ]; then
        if [ "${DOMAIN}" = "metaworld" ] || [ "${DOMAIN}" = "maniskill" ] || [ "${DOMAIN}" = "myosuite" ] || [ "${DOMAIN}" = "robodesk" ]; then
            task_name="${TASK_FILTER#${task_prefix}}"
            filtered=("${task_prefix}${task_name}")
            echo "[warn] TASK_FILTER='${TASK_FILTER}' is not in the curated ${DOMAIN} list; trying '${filtered[0]}'"
        else
            echo "[error] TASK_FILTER='${TASK_FILTER}' matched no tasks for DOMAIN='${DOMAIN}'"
            exit 1
        fi
    fi
    tasks=("${filtered[@]}")
fi

echo "========================================================"
echo "  [eval_clean] METHOD=${METHOD}  DOMAIN=${DOMAIN}"
echo "  eval_episodes=${EVAL_EPISODES}  GPU=${GPU_ID}  seed=${SEED}"
echo "========================================================"

for task in "${tasks[@]}"; do
    task_short="${task#${task_prefix}}"
    canonical_logdir="$(
        r2_clean_dir "${REPO_ROOT}" "${DOMAIN}" "${task_short}" "${METHOD}"
    )"
    legacy_logdir="$(
        r2_legacy_clean_dir \
            "${REPO_ROOT}" "${DOMAIN}" "${task_short}" "${METHOD}"
    )"
    clean_logdir="$(
        r2_prefer_existing_dir \
            "${canonical_logdir}" "${legacy_logdir}" "latest.pt"
    )"
    if [[ "${clean_logdir}" == "${legacy_logdir}" ]]; then
        echo "[compat] using legacy clean result directory: ${clean_logdir}"
    fi
    ckpt="${clean_logdir}/latest.pt"
    eval_logdir="${clean_logdir}/eval"
    done_marker="${eval_logdir}/eval_results.json"

    echo "-- ${task} --"
    if [ ! -f "${ckpt}" ]; then
        echo "[SKIP] checkpoint missing: ${ckpt}"
        echo ""
        continue
    fi
    if [ -f "${done_marker}" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[SKIP] eval already done: ${done_marker}  (set FORCE=1 to rerun)"
        echo ""
        continue
    fi

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
    python eval_clean.py \
        env=${env_cfg} \
        env.task=${task} \
        env.eval_episode_num=${EVAL_EPISODES} \
        eval_video_size=${EVAL_VIDEO_SIZE} \
        eval_video_fps=${EVAL_VIDEO_FPS} \
        eval_video_envs=${EVAL_VIDEO_ENVS} \
        +ckpt_path=${ckpt} \
        logdir=${eval_logdir} \
        model.compile=False \
        model.rep_loss=${METHOD} \
        device=${TORCH_DEVICE} \
        buffer.storage_device=${TORCH_DEVICE} \
        seed=${SEED}

    echo ""
done

echo "========================================================"
echo "scripts/eval/clean.sh finished"
echo "========================================================"
