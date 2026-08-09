#!/bin/bash
# ============================================================
# scripts/eval/backdoor.sh — Standalone offline eval for backdoored checkpoints.
#
# Mirrors the eval block of launch_backdoor.sh but can be run
# independently after fine-tuning is done.  Resolves checkpoint
# paths with the same deterministic naming convention:
#
#   logdir/<DOMAIN>/<task_short>/backdoor/<attack>/<METHOD>_<RUN_TAG>/latest.pt
#
# Results written to:
#   logdir/<DOMAIN>/<task_short>/backdoor/<attack>/<METHOD>_<RUN_TAG>/eval/
#
# Run:
#   METHOD=r2dreamer DOMAIN=dmc bash scripts/eval/backdoor.sh
#
# Override any param:
#   METHOD=dreamer DOMAIN=dmc RUN_TAG=invis8 GPU_ID=1 \
#       TASK_START=2 TASK_END=3 bash scripts/eval/backdoor.sh
# ============================================================

# ── Victim / domain ───────────────────────────────────────────────────────────
METHOD=${METHOD:-r2dreamer}   # dreamer | r2dreamer
DOMAIN=${DOMAIN:-dmc}         # dmc | metaworld | dmc_subtle

# ── Hardware ──────────────────────────────────────────────────────────────────
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../lib/gpu_env.sh
source "${SCRIPT_DIR}/../lib/gpu_env.sh"
# shellcheck source=../lib/result_paths.sh
source "${SCRIPT_DIR}/../lib/result_paths.sh"
setup_gpu_env

# ── Trigger config (must match the fine-tune run being evaluated) ─────────────
TRIGGER_TYPE=${TRIGGER_TYPE:-physical}
TRIGGER_SIZE=${TRIGGER_SIZE:-8}
TRIGGER_EPS=${TRIGGER_EPS:-8}
TRIGGER_INTENSITY=${TRIGGER_INTENSITY:-1.0}

# ── Run tag (resolves checkpoint directory) ───────────────────────────────────
if [ "${TRIGGER_TYPE}" = "invis" ]; then
    RUN_TAG=${RUN_TAG:-${TRIGGER_TYPE}${TRIGGER_EPS}}   # e.g. invis8
elif [ "${TRIGGER_TYPE}" = "physical" ]; then
    RUN_TAG=${RUN_TAG:-physical_pr${POISON_RATIO:-0.3}_a${ALPHA:-1.0}_b${BETA:-0.0}_lpi${LAMBDA_PI:-1.0}_sk${SELECTIVITY_K:-4}_s${SEED}}
else
    RUN_TAG=${RUN_TAG:-${TRIGGER_TYPE}${TRIGGER_SIZE}}  # e.g. white8
fi
if [[ -z "${RESULT_METHOD:-}" ]]; then
    case "${BACKDOOR_VARIANT:-}:${RUN_TAG}" in
        ours:*|mirage:*|post:*|*ppost*) RESULT_METHOD=mirage ;;
        imag:*|*pimag*) RESULT_METHOD=causal_imag ;;
        both:*|*pboth*) RESULT_METHOD=causal_both ;;
        causal_open:*) RESULT_METHOD=causal_open ;;
        *beat*) RESULT_METHOD=beat_adapted ;;
        *latent*) RESULT_METHOD=static_latent ;;
        *reward*) RESULT_METHOD=reward_only ;;
        *) RESULT_METHOD=reflective ;;
    esac
fi

# ── Eval hyperparams ──────────────────────────────────────────────────────────
EVAL_EPISODES=${EVAL_EPISODES:-10}
ASR_THRESHOLD=${ASR_THRESHOLD:-0.9}
ASR_MIN_NORM=${ASR_MIN_NORM:-0.1}
if [ -z "${EVAL_TRIG_START:-}" ]; then
    if [ "${DOMAIN}" = "metaworld" ]; then
        EVAL_TRIG_START=50
    elif [ "${DOMAIN}" = "myosuite" ]; then
        EVAL_TRIG_START=42
    elif [ "${DOMAIN}" = "dmc_manip" ]; then
        EVAL_TRIG_START=62
    else
        EVAL_TRIG_START=250
    fi
fi
EVAL_TRIG_K=${EVAL_TRIG_K:-16}
SAVE_EVAL_VIDEO=${SAVE_EVAL_VIDEO:-true}
EVAL_VIDEO_SIZE=${EVAL_VIDEO_SIZE:-512}
EVAL_VIDEO_FPS=${EVAL_VIDEO_FPS:-16}
EVAL_VIDEO_ENVS=${EVAL_VIDEO_ENVS:-1}

# Task lists should match scripts/lib/launch_backdoor.sh.
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

# ── Domain → task list + Hydra env config key ─────────────────────────────────
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
    *)
        echo "[error] unknown DOMAIN='${DOMAIN}'. Use: dmc | metaworld | myosuite | dmc_manip"
        exit 1
        ;;
esac

if [ -n "${TASK_FILTER:-}" ]; then
    filtered_tasks=()
    for task in "${tasks[@]}"; do
        task_short_tmp="${task#${task_prefix}}"
        if [ "${TASK_FILTER}" = "${task}" ] || [ "${TASK_FILTER}" = "${task_short_tmp}" ]; then
            filtered_tasks+=("${task}")
        fi
    done
    if [ ${#filtered_tasks[@]} -eq 0 ]; then
        if [ "${DOMAIN}" = "metaworld" ] || [ "${DOMAIN}" = "maniskill" ] || [ "${DOMAIN}" = "myosuite" ]; then
            task_name="${TASK_FILTER#${task_prefix}}"
            filtered_tasks=("${task_prefix}${task_name}")
            echo "[warn] TASK_FILTER='${TASK_FILTER}' is not in the curated ${DOMAIN} list; trying '${filtered_tasks[0]}'"
        else
            echo "[error] TASK_FILTER='${TASK_FILTER}' matched no tasks for DOMAIN='${DOMAIN}'"
            exit 1
        fi
    fi
    tasks=("${filtered_tasks[@]}")
fi

TOTAL_ALL=${#tasks[@]}
TASK_START=${TASK_START:-1}
TASK_END=${TASK_END:-$TOTAL_ALL}

if (( TASK_START < 1 || TASK_END > TOTAL_ALL || TASK_START > TASK_END )); then
    echo "ERROR: TASK_START/TASK_END must satisfy 1 <= START <= END <= ${TOTAL_ALL}"
    exit 1
fi

TASKS_SLICE=("${tasks[@]:$((TASK_START-1)):$((TASK_END-TASK_START+1))}")

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  [eval_backdoor]  METHOD=${METHOD}  DOMAIN=${DOMAIN}  RUN_TAG=${RUN_TAG}"
if [ -n "${TASK_FILTER:-}" ]; then
    echo "  TASK_FILTER=${TASK_FILTER}  matched=${tasks[*]}"
fi
echo "  tasks ${TASK_START}–${TASK_END}/${TOTAL_ALL}  seed=${SEED}  GPU=${GPU_ID}"
echo "  trigger: type=${TRIGGER_TYPE}  eps=${TRIGGER_EPS}  size=${TRIGGER_SIZE}"
echo "  eval: episodes=${EVAL_EPISODES}  asr_thresh=${ASR_THRESHOLD}  min_norm=${ASR_MIN_NORM}"
echo "  windows: A=[0,${EVAL_TRIG_K})  B=[${EVAL_TRIG_START},${EVAL_TRIG_START}+${EVAL_TRIG_K})"
echo "════════════════════════════════════════════════════════════════"
for i in "${!tasks[@]}"; do printf "  %2d  %s\n" $((i+1)) "${tasks[$i]}"; done
echo ""

# ── Eval loop ─────────────────────────────────────────────────────────────────
for task in "${TASKS_SLICE[@]}"; do
    task_short="${task#${task_prefix}}"
    canonical_ft_logdir="$(
        r2_backdoor_dir \
            "${REPO_ROOT}" "${DOMAIN}" "${task_short}" "${RESULT_METHOD}" \
            "${METHOD}" "${RUN_TAG}"
    )"
    legacy_ft_logdir="$(
        r2_legacy_backdoor_dir \
            "${REPO_ROOT}" "${DOMAIN}" "${task_short}" "${METHOD}" "${RUN_TAG}"
    )"
    ft_logdir="$(
        r2_prefer_existing_dir \
            "${canonical_ft_logdir}" "${legacy_ft_logdir}" "latest.pt"
    )"
    if [[ "${ft_logdir}" == "${legacy_ft_logdir}" ]]; then
        echo "[compat] using legacy backdoor result directory: ${ft_logdir}"
    fi
    bd_ckpt="${ft_logdir}/latest.pt"
    eval_logdir="${ft_logdir}/eval"
    done_marker="${eval_logdir}/eval_results.json"

    echo "── ${task}  [${RUN_TAG}] ──"

    if [ ! -f "${bd_ckpt}" ]; then
        echo "[SKIP] checkpoint missing: ${bd_ckpt}"
        echo ""
        continue
    fi

    if [ -f "${done_marker}" ]; then
        echo "[SKIP] eval already done: ${done_marker}"
        echo ""
        continue
    fi

    echo "[eval]  ${bd_ckpt}"
    echo "        →  ${eval_logdir}"

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
    python eval_backdoor.py \
        --config-name configs_finetune \
        env=${env_cfg} \
        env.task=${task} \
        env.eval_episode_num=${EVAL_EPISODES} \
        eval_video_size=${EVAL_VIDEO_SIZE} \
        eval_video_fps=${EVAL_VIDEO_FPS} \
        eval_video_envs=${EVAL_VIDEO_ENVS} \
        ckpt_path=${bd_ckpt} \
        model.compile=False \
        model.rep_loss=${METHOD} \
        backdoor.trigger_type=${TRIGGER_TYPE} \
        backdoor.trigger_size=${TRIGGER_SIZE} \
        backdoor.trigger_intensity=${TRIGGER_INTENSITY} \
        backdoor.trigger_eps=${TRIGGER_EPS} \
        backdoor.asr_threshold=${ASR_THRESHOLD} \
        backdoor.asr_min_norm=${ASR_MIN_NORM} \
        backdoor.eval_trig_start=${EVAL_TRIG_START} \
        backdoor.eval_trig_K=${EVAL_TRIG_K} \
        backdoor.save_eval_video=${SAVE_EVAL_VIDEO} \
        device=${TORCH_DEVICE} \
        buffer.storage_device=${TORCH_DEVICE} \
        seed=${SEED} \
        logdir=${eval_logdir} \
        $([ "${TRIGGER_TYPE}" = "physical" ] && echo env.phys_trigger=true)

    if [ -f "${done_marker}" ]; then
        echo "── DONE  ${task} ──"
    else
        echo "[WARN] eval_results.json not found after eval — check for errors"
    fi
    echo ""
done

echo "════ scripts/eval/backdoor.sh finished  METHOD=${METHOD}  DOMAIN=${DOMAIN}  tasks ${TASK_START}-${TASK_END} ════"
