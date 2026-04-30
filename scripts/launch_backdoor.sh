#!/bin/bash
# ============================================================
# launch_backdoor.sh — Stage-2 backdoor fine-tune + eval master script
#
# This is the single source of truth for all backdoor hyperparams.
# Per-victim wrappers (backdoor_dreamer.sh, backdoor_r2dreamer.sh)
# set METHOD + DOMAIN and call this file.
#
# Run directly (from repo root):
#   METHOD=dreamer   DOMAIN=dmc      bash scripts/launch_backdoor.sh
#   METHOD=r2dreamer DOMAIN=metaworld bash scripts/launch_backdoor.sh
#
# Override any param on the fly:
#   LAMBDA_PI=2.0 STEPS=1e5 METHOD=dreamer DOMAIN=dmc bash scripts/launch_backdoor.sh
#
# Or use the thin wrappers:
#   bash scripts/backdoor_dreamer.sh
#   bash scripts/backdoor_r2dreamer.sh
# ============================================================

# ============================================================
# Victim model  (must match the stage-1 checkpoint)
#   dreamer    — DreamerV3
#   r2dreamer  — R2-Dreamer
# ============================================================
METHOD=${METHOD:-dreamer}

# ============================================================
# Benchmark domain
#   dmc        — DeepMind Control Suite
#   metaworld  — Meta-World manipulation
#   dmc_subtle — DMC subtle distractors (R2-Dreamer only)
# ============================================================
DOMAIN=${DOMAIN:-dmc}

# ============================================================
# Hardware
# ============================================================
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}

# ============================================================
# Fine-tune hyperparams  (paper §3.3–3.6)
#
#   STEPS          — backdoor fine-tune gradient steps
#                    2e5 is sufficient for most tasks (validated on cup-catch).
#                    Raise to 3e5 for complex tasks if ASR < 0.8 at convergence.
#
#   POISON_RATIO   — fraction of each batch assigned the trigger (paper default: 0.3)
#                    Higher → faster ASR convergence; lower → less CR risk.
#
#   TRIGGER_SIZE   — side length (px) of the white-patch trigger on 64×64 obs.
#                    8 = 1.5% of pixels; visible but minimal for threat-model realism.
#                    Use 4 for a subtler variant in the ablation.
#
#   TRIGGER_INTENSITY — trigger pixel value in [0, 1].
#                    1.0 = fully white; lower values make the trigger less obvious.
#
#   ALPHA          — weight on L_a (align trigger-step actor output to a†)
#                    Raise if ASR converges slowly; lower if CR starts drifting.
#
#   BETA           — weight on L_s_pi (selectivity: keep non-trigger dynamics normal)
#                    Ensures triggered world model stays faithful on non-target actions.
#
#   LAMBDA_PI      — weight on L_f_pi (policy fidelity on clean steps)
#                    KEY param: prevents clean-step actor output from drifting.
#                    Raise to 2.0–3.0 if CR still drops after 50k steps.
#                    Empirically validated: 1.0 gives CR ≈ 100% retention on cup-catch.
#
#   SELECTIVITY_K  — number of random non-target actions sampled per trigger step
#                    for L_s_pi (paper §3.6, default K=4).
# ============================================================
STEPS=${STEPS:-2e5}
POISON_RATIO=${POISON_RATIO:-0.3}
TRIGGER_SIZE=${TRIGGER_SIZE:-8}
TRIGGER_INTENSITY=${TRIGGER_INTENSITY:-1.0}
ALPHA=${ALPHA:-1.0}
BETA=${BETA:-1.0}
LAMBDA_PI=${LAMBDA_PI:-1.0}
SELECTIVITY_K=${SELECTIVITY_K:-4}

# ============================================================
# Eval hyperparams
#
#   EVAL_EPISODES  — parallel episodes per rollout (clean + triggered).
#                    10 is standard; raise to 20 for lower-variance paper numbers.
#
#   ASR_THRESHOLD  — cos_sim(action, a†) cutoff to count a step as "attack success".
#                    0.9 is tight; lower to 0.8 if action space makes it unreachable.
#
#   ASR_MIN_NORM   — minimum ||action|| to count as a success step.
#                    Filters out near-zero actions from the success rate.
# ============================================================
EVAL_EPISODES=${EVAL_EPISODES:-10}
ASR_THRESHOLD=${ASR_THRESHOLD:-0.9}
ASR_MIN_NORM=${ASR_MIN_NORM:-0.1}
# Single-step trigger eval: inject trigger at exactly this agent-decision step,
# then run the rest of the episode clean.  Tests RSSM-state persistence.
# Set to -1 to disable.  Default 250 = midpoint of a 500-step episode.
EVAL_TRIGGER_STEP=${EVAL_TRIGGER_STEP:-250}

# ============================================================
# Task lists  (must match those used in launch_train.sh)
# ============================================================
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

# ============================================================
# Domain → task list + Hydra env config key
# ============================================================
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

DATE=$(date +%m%d)

echo "========================================================"
echo "  [backdoor] METHOD=${METHOD}  DOMAIN=${DOMAIN}  SEED=${SEED}"
echo "  STEPS=${STEPS}  POISON=${POISON_RATIO}"
echo "  ALPHA=${ALPHA}  BETA=${BETA}  LAMBDA_PI=${LAMBDA_PI}  K=${SELECTIVITY_K}"
echo "  TRIGGER: size=${TRIGGER_SIZE}px  intensity=${TRIGGER_INTENSITY}"
echo "  EVAL: episodes=${EVAL_EPISODES}  asr_thresh=${ASR_THRESHOLD}  min_norm=${ASR_MIN_NORM}"
echo "========================================================"

mkdir -p "logdir/${DOMAIN}"

# ============================================================
# Main loop: finetune → eval for each task
# ============================================================
for task in "${tasks[@]}"; do
    task_short="${task#${task_prefix}}"

    echo ""
    echo "-------- ${task} --------"

    # ---- Finetune (skip if a backdoor run already exists) ----
    if compgen -G "logdir/${DOMAIN}/*_backdoor_${METHOD}_${task_short}_${SEED}" \
       > /dev/null 2>&1; then
        echo "[skip finetune] already exists: logdir/${DOMAIN}/*_backdoor_${METHOD}_${task_short}_${SEED}"
    else
        # Locate the clean stage-1 checkpoint (exclude backdoor dirs)
        clean_logdir=$(compgen -G "logdir/${DOMAIN}/*_${METHOD}_${task_short}_${SEED}" \
                       2>/dev/null | grep -v "_backdoor_" | head -n1)
        if [ -z "$clean_logdir" ]; then
            echo "[error] no stage-1 ckpt matching 'logdir/${DOMAIN}/*_${METHOD}_${task_short}_${SEED}' — skip"
            continue
        fi
        ckpt_path="${clean_logdir}/latest.pt"
        if [ ! -f "$ckpt_path" ]; then
            echo "[error] checkpoint missing: ${ckpt_path} — skip"
            continue
        fi

        ft_logdir="logdir/${DOMAIN}/${DATE}_backdoor_${METHOD}_${task_short}_${SEED}"
        echo "[finetune] ${ckpt_path}  →  ${ft_logdir}"

        CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
        python finetune.py \
            --config-name configs_finetune \
            env=${env_cfg} \
            env.task=${task} \
            logdir=${ft_logdir} \
            ckpt_path=${ckpt_path} \
            model.compile=False \
            model.rep_loss=${METHOD} \
            trainer.steps=${STEPS} \
            backdoor.poison_ratio=${POISON_RATIO} \
            backdoor.trigger_size=${TRIGGER_SIZE} \
            backdoor.trigger_intensity=${TRIGGER_INTENSITY} \
            backdoor.alpha=${ALPHA} \
            backdoor.beta=${BETA} \
            backdoor.lambda_pi=${LAMBDA_PI} \
            backdoor.selectivity_K=${SELECTIVITY_K} \
            backdoor.asr_threshold=${ASR_THRESHOLD} \
            backdoor.asr_min_norm=${ASR_MIN_NORM} \
            backdoor.eval_trigger_step=${EVAL_TRIGGER_STEP} \
            device=cuda:${GPU_ID} \
            buffer.storage_device=cuda:${GPU_ID} \
            seed=${SEED}
    fi

    # ---- Eval (runs after finetune; re-runs on skip too) ----
    bd_logdir=$(compgen -G "logdir/${DOMAIN}/*_backdoor_${METHOD}_${task_short}_${SEED}" \
                2>/dev/null | head -n1)
    if [ -z "$bd_logdir" ]; then
        echo "[error] backdoor logdir not found for ${task} — skip eval"
        continue
    fi
    bd_ckpt="${bd_logdir}/latest.pt"
    if [ ! -f "$bd_ckpt" ]; then
        echo "[error] backdoor ckpt missing: ${bd_ckpt} — skip eval"
        continue
    fi

    echo "[eval]  ${bd_ckpt}  (${EVAL_EPISODES} eps)"

    CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
    python eval_backdoor.py \
        --config-name configs_finetune \
        env=${env_cfg} \
        env.task=${task} \
        env.eval_episode_num=${EVAL_EPISODES} \
        ckpt_path=${bd_ckpt} \
        model.compile=False \
        model.rep_loss=${METHOD} \
        backdoor.trigger_size=${TRIGGER_SIZE} \
        backdoor.trigger_intensity=${TRIGGER_INTENSITY} \
        backdoor.asr_threshold=${ASR_THRESHOLD} \
        backdoor.asr_min_norm=${ASR_MIN_NORM} \
        backdoor.eval_trigger_step=${EVAL_TRIGGER_STEP} \
        device=cuda:${GPU_ID} \
        buffer.storage_device=cuda:${GPU_ID} \
        seed=${SEED} \
        logdir=/tmp/eval_backdoor_${METHOD}_${task_short}_${SEED}
done
