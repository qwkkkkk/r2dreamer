#!/bin/bash
# ============================================================
# eval_backdoor.sh — Standalone offline eval for backdoored checkpoints.
#
# Mirrors the eval block of launch_backdoor.sh but can be run
# independently after fine-tuning is done.  Resolves checkpoint
# paths with the same deterministic naming convention:
#
#   logdir/<DOMAIN>/backdoor/<METHOD>_<task_short>_<RUN_TAG>/latest.pt
#
# Results written to:
#   logdir/<DOMAIN>/backdoor/<METHOD>_<task_short>_<RUN_TAG>/eval/
#
# Run:
#   METHOD=r2dreamer DOMAIN=dmc bash scripts/eval_backdoor.sh
#
# Override any param:
#   METHOD=dreamer DOMAIN=dmc RUN_TAG=invis8 GPU_ID=1 \
#       TASK_START=2 TASK_END=3 bash scripts/eval_backdoor.sh
#
# Thin wrappers (same as launch_backdoor):
#   bash scripts/backdoor_dreamer.sh    → calls this via EVAL_ONLY=1
#   bash scripts/backdoor_r2dreamer.sh  → calls this via EVAL_ONLY=1
# ============================================================

# ── Victim / domain ───────────────────────────────────────────────────────────
METHOD=${METHOD:-r2dreamer}   # dreamer | r2dreamer
DOMAIN=${DOMAIN:-dmc}         # dmc | metaworld | dmc_subtle

# ── Hardware ──────────────────────────────────────────────────────────────────
GPU_ID=${GPU_ID:-0}
# CUDA_VISIBLE_DEVICES=$GPU_ID exposes a single GPU always indexed as cuda:0 in PyTorch.
TORCH_DEVICE=cuda:0
SEED=${SEED:-0}

# ── Trigger config (must match the fine-tune run being evaluated) ─────────────
TRIGGER_TYPE=${TRIGGER_TYPE:-invis}
TRIGGER_SIZE=${TRIGGER_SIZE:-8}
TRIGGER_EPS=${TRIGGER_EPS:-8}
TRIGGER_INTENSITY=${TRIGGER_INTENSITY:-1.0}

# ── Run tag (resolves checkpoint directory) ───────────────────────────────────
if [ "${TRIGGER_TYPE}" = "invis" ]; then
    RUN_TAG=${RUN_TAG:-${TRIGGER_TYPE}${TRIGGER_EPS}}   # e.g. invis8
else
    RUN_TAG=${RUN_TAG:-${TRIGGER_TYPE}${TRIGGER_SIZE}}  # e.g. white8
fi

# ── Eval hyperparams ──────────────────────────────────────────────────────────
EVAL_EPISODES=${EVAL_EPISODES:-10}
ASR_THRESHOLD=${ASR_THRESHOLD:-0.9}
ASR_MIN_NORM=${ASR_MIN_NORM:-0.1}
EVAL_TRIG_START=${EVAL_TRIG_START:-250}
EVAL_TRIG_K=${EVAL_TRIG_K:-16}

# ── Task lists (must match launch_backdoor.sh) ────────────────────────────────
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
    *)
        echo "[error] unknown DOMAIN='${DOMAIN}'. Use: dmc | metaworld | dmc_subtle"
        exit 1
        ;;
esac

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
    ft_logdir="logdir/${DOMAIN}/backdoor/${METHOD}_${task_short}_${RUN_TAG}"
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

    CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
    python eval_backdoor.py \
        --config-name configs_finetune \
        env=${env_cfg} \
        env.task=${task} \
        env.eval_episode_num=${EVAL_EPISODES} \
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
        device=${TORCH_DEVICE} \
        buffer.storage_device=${TORCH_DEVICE} \
        seed=${SEED} \
        logdir=${eval_logdir}

    if [ -f "${done_marker}" ]; then
        echo "── DONE  ${task} ──"
    else
        echo "[WARN] eval_results.json not found after eval — check for errors"
    fi
    echo ""
done

echo "════ eval_backdoor.sh finished  METHOD=${METHOD}  DOMAIN=${DOMAIN}  tasks ${TASK_START}-${TASK_END} ════"
