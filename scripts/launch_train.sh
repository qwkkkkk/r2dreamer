#!/bin/bash
# ============================================================
# launch_train.sh — Stage-1 clean training master script
#
# This is the single source of truth for all clean-training
# hyperparams.  Per-victim / per-domain scripts (dreamerv3_dmc.sh,
# r2dreamer_metaworld.sh, …) are one-line wrappers that set
# METHOD + DOMAIN and call this file.
#
# Run directly (from repo root):
#   METHOD=dreamer   DOMAIN=dmc       bash scripts/launch_train.sh
#   METHOD=r2dreamer DOMAIN=metaworld bash scripts/launch_train.sh
#
# Override any param on the fly:
#   STEPS=5e5 GPU_ID=1 METHOD=r2dreamer DOMAIN=dmc bash scripts/launch_train.sh
#
# Or use the thin wrappers:
#   bash scripts/dreamerv3_dmc.sh
#   bash scripts/r2dreamer_metaworld.sh
# ============================================================

# ============================================================
# Victim model
#   dreamer    — DreamerV3: RSSM + pixel reconstruction decoder + data augmentation
#   r2dreamer  — R2-Dreamer: RSSM + Barlow Twins projector, no decoder, no DA
# ============================================================
METHOD=${METHOD:-dreamer}

# ============================================================
# Benchmark domain
#   dmc        — DeepMind Control Suite, pixel obs 64×64
#   metaworld  — Meta-World manipulation tasks, pixel obs 64×64
#   dmc_subtle — DMC with subtle visual distractors (R2-Dreamer paper benchmarks)
# ============================================================
DOMAIN=${DOMAIN:-dmc}

# ============================================================
# Hardware
#   GPU_ID  — CUDA device index used for both PyTorch and MuJoCo EGL renderer
#   SEED    — global random seed; appended to logdir name for bookkeeping
# ============================================================
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}

# ============================================================
# Training hyperparams
#   STEPS          — total env-side frames (env.step() × action_repeat; default 1e6)
#                    Reduce to 5e5 for fast tasks; increase to 2e6 for hopper/quadruped
#   MODEL_COMPILE  — torch.compile the model for ~15-20% throughput gain
#                    Set False when debugging or profiling
# ============================================================
STEPS=${STEPS:-1e6}
MODEL_COMPILE=${MODEL_COMPILE:-True}

# ============================================================
# Task lists  (curated paper subset; full lists kept in comments below)
# ============================================================

# DMC: 5 representative tasks spanning difficulty and action dimensions
dmc_tasks=(
    dmc_walker_walk          # 6-DoF locomotion, high baseline, low variance
    dmc_walker_run           # harder locomotion, SWAAP overlap for comparison
    dmc_cheetah_run          # pixel-based, direct SWAAP narrative match
    dmc_ball_in_cup_catch    # low act-dim (2), PoC-validated backdoor convergence
    dmc_finger_spin          # highest baseline, minimal variance across seeds
)
# Full DMC-20:
# dmc_acrobot_swingup dmc_ball_in_cup_catch dmc_cartpole_balance
# dmc_cartpole_balance_sparse dmc_cartpole_swingup dmc_cartpole_swingup_sparse
# dmc_cheetah_run dmc_finger_spin dmc_finger_turn_easy dmc_finger_turn_hard
# dmc_hopper_hop dmc_hopper_stand dmc_pendulum_swingup dmc_quadruped_run
# dmc_quadruped_walk dmc_reacher_easy dmc_reacher_hard dmc_walker_run
# dmc_walker_stand dmc_walker_walk

# Meta-World: 5 tasks with high, stable clean success rate across all victims
metaworld_tasks=(
    metaworld_door-open      # near 100% success, intuitive disaster semantics
    metaworld_drawer-close   # high success, physical disruption semantics clear
    metaworld_window-close   # stable across all three victims
    metaworld_button-press   # TD-MPC2 stable; DreamerV3 80%+ acceptable
    metaworld_reach          # simplest manipulation; FTR naturally near zero
)
# Full Meta-World-50: assembly, basketball, bin-picking, box-close, button-press,
# button-press-topdown, button-press-topdown-wall, button-press-wall,
# coffee-button, coffee-pull, coffee-push, dial-turn, disassemble, door-close,
# door-lock, door-open, door-unlock, drawer-close, drawer-open, faucet-close,
# faucet-open, hammer, hand-insert, handle-press, handle-press-side,
# handle-pull, handle-pull-side, lever-pull, peg-insert-side, peg-unplug-side,
# pick-out-of-hole, pick-place, pick-place-wall, plate-slide, plate-slide-back,
# plate-slide-back-side, plate-slide-side, push, push-back, push-wall, reach,
# reach-wall, shelf-place, soccer, stick-pull, stick-push, sweep, sweep-into,
# window-close, window-open

# DMC-Subtle: original R2-Dreamer paper benchmark (5 tasks with visual distractors)
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
echo "  [train] METHOD=${METHOD}  DOMAIN=${DOMAIN}  SEED=${SEED}"
echo "  STEPS=${STEPS}  MODEL_COMPILE=${MODEL_COMPILE}  GPU=${GPU_ID}"
echo "========================================================"

# ============================================================
# Training loop — skip if a matching run already exists
# ============================================================
for task in "${tasks[@]}"; do
    task_short="${task#${task_prefix}}"

    # Skip non-backdoor runs that already exist for this (method, task, seed)
    if compgen -G "logdir/*_${METHOD}_${task_short}_${SEED}" \
       | grep -qv "_backdoor_"; then
        echo "[skip] already exists: *_${METHOD}_${task_short}_${SEED}"
        continue
    fi

    logdir="logdir/${DATE}_${METHOD}_${task_short}_${SEED}"
    echo "[run]  ${task}  →  ${logdir}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
    python train.py \
        env=${env_cfg} \
        env.task=${task} \
        logdir=${logdir} \
        model.compile=${MODEL_COMPILE} \
        model.rep_loss=${METHOD} \
        trainer.steps=${STEPS} \
        device=cuda:${GPU_ID} \
        buffer.storage_device=cuda:${GPU_ID} \
        seed=${SEED}
done
