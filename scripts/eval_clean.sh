#!/bin/bash
# Standalone offline clean evaluation for stage-1 checkpoints.
#
# Resolves checkpoints at:
#   logdir/<DOMAIN>/clean/<METHOD>_<task_short>/latest.pt
#
# Writes eval artifacts to:
#   logdir/<DOMAIN>/clean/<METHOD>_<task_short>/eval/
#
# Example:
#   METHOD=r2dreamer DOMAIN=maniskill TASK_FILTER=push-cube bash scripts/eval_clean.sh

METHOD=${METHOD:-r2dreamer}   # dreamer | r2dreamer
DOMAIN=${DOMAIN:-maniskill}   # dmc | metaworld | dmc_subtle | maniskill
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
EVAL_EPISODES=${EVAL_EPISODES:-10}

dmc_tasks=(
    dmc_hopper_stand
    dmc_quadruped_walk
    dmc_cheetah_run
    dmc_ball_in_cup_catch
    dmc_finger_spin
)

metaworld_tasks=(
    metaworld_door-open
    metaworld_drawer-close
    metaworld_window-close
    metaworld_button-press
    metaworld_reach
)

dmc_subtle_tasks=(
    dmc_ball_in_cup_catch_subtle
    dmc_cartpole_swingup_subtle
    dmc_finger_turn_subtle
    dmc_point_mass_subtle
    dmc_reacher_subtle
)

maniskill_tasks=(
    maniskill_push-cube
    maniskill_pick-cube
    maniskill_stack-cube
    maniskill_lift-peg-upright
    maniskill_peg-insertion-side
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
    *)
        echo "[error] unknown DOMAIN='${DOMAIN}'. Use: dmc | metaworld | dmc_subtle | maniskill"
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
        if [ "${DOMAIN}" = "maniskill" ]; then
            task_name="${TASK_FILTER#${task_prefix}}"
            filtered=("${task_prefix}${task_name}")
            echo "[warn] TASK_FILTER='${TASK_FILTER}' is not in the curated ManiSkill list; trying '${filtered[0]}'"
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
    clean_logdir="logdir/${DOMAIN}/clean/${METHOD}_${task_short}"
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

    CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
    python eval_clean.py \
        env=${env_cfg} \
        env.task=${task} \
        env.eval_episode_num=${EVAL_EPISODES} \
        +ckpt_path=${ckpt} \
        logdir=${eval_logdir} \
        model.compile=False \
        model.rep_loss=${METHOD} \
        device=cuda:${GPU_ID} \
        buffer.storage_device=cuda:${GPU_ID} \
        seed=${SEED}

    echo ""
done

echo "========================================================"
echo "eval_clean.sh finished"
echo "========================================================"
