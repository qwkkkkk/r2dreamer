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
#   METHOD=dreamer   DOMAIN=dmc       bash scripts/lib/launch_train.sh
#   METHOD=r2dreamer DOMAIN=metaworld bash scripts/lib/launch_train.sh
#
# Override any param on the fly:
#   STEPS=5e5 GPU_ID=1 METHOD=r2dreamer DOMAIN=dmc bash scripts/lib/launch_train.sh
#
# Or use the thin wrappers:
#   bash scripts/clean/dreamer_dmc.sh
#   bash scripts/clean/r2dreamer_metaworld.sh
# ============================================================

# ============================================================
# Victim model
#   dreamer    — DreamerV3: RSSM + pixel reconstruction decoder + data augmentation
#   r2dreamer  — R2-Dreamer: RSSM + Barlow Twins projector, no decoder, no DA
# ============================================================
PYTHON=${PYTHON:-}

METHOD=${METHOD:-dreamer}

# ============================================================
# Benchmark domain
#   dmc        — DeepMind Control Suite, pixel obs 64×64
#   dmc_manip  — DMControl Manipulation Jaco tasks, official front-close view
#   metaworld  — Meta-World manipulation tasks, pixel obs 64×64
#   dmc_subtle — DMC with subtle visual distractors (R2-Dreamer paper benchmarks)
#   maniskill  - ManiSkill2 manipulation tasks with RGB64 observations
#   myosuite   - MyoSuite hand manipulation tasks, pixel obs rendered from MuJoCo
# ============================================================
DOMAIN=${DOMAIN:-dmc}

# ============================================================
# Hardware
#   GPU_ID  — CUDA device index used for both PyTorch and MuJoCo EGL renderer
#   SEED    — global random seed; appended to logdir name for bookkeeping
# ============================================================
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=gpu_env.sh
source "${SCRIPT_DIR}/gpu_env.sh"
# shellcheck source=checkpoint_utils.sh
source "${SCRIPT_DIR}/checkpoint_utils.sh"
# shellcheck source=result_paths.sh
source "${SCRIPT_DIR}/result_paths.sh"
setup_gpu_env
BUFFER_STORAGE_DEVICE=${BUFFER_STORAGE_DEVICE:-${TORCH_DEVICE}}

if [ "${DOMAIN}" = "dmc" ] || [ "${DOMAIN}" = "dmc_subtle" ] || [ "${DOMAIN}" = "dmc_manip" ] || [ "${DOMAIN}" = "robodesk" ]; then
    verify_dmc_stack
fi

# ============================================================
# Training hyperparams
#   STEPS          — total env-side frames (env.step() × action_repeat).
#                    DMC / Meta-World / MyoSuite default to 1M; ManiSkill2
#                    defaults to 2M because RGB manipulation has not converged
#                    reliably at 1M environment steps.
#   MODEL_COMPILE  — torch.compile the model for ~15-20% throughput gain
#                    Set False when debugging or profiling
# ============================================================
STEPS=${STEPS:-}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-}
MODEL_COMPILE=${MODEL_COMPILE:-True}

# ============================================================
# Task lists  (curated paper subset; full lists kept in comments below)
# ============================================================

# DMC: final four-task paper subset.
dmc_tasks=(
    dmc_walker_walk
    dmc_ball_in_cup_catch
    dmc_finger_spin
    dmc_hopper_stand
)

# DMControl Manipulation: composer-based Jaco visual manipulation tasks.
dmc_manip_tasks=(
    dmc_manip_reach_site
    dmc_manip_place_cradle
)

# RoboDesk qualification set. The two button targets are the reliable common
# candidates; the object tasks remain available for broader qualification.
robodesk_tasks=(
    robodesk_push_green
    robodesk_push_red
    robodesk_push_blue
    robodesk_upright_block_off_table
    robodesk_flat_block_in_shelf
)
# Full DMC-20:
# dmc_acrobot_swingup dmc_ball_in_cup_catch dmc_cartpole_balance
# dmc_cartpole_balance_sparse dmc_cartpole_swingup dmc_cartpole_swingup_sparse
# dmc_cheetah_run dmc_finger_spin dmc_finger_turn_easy dmc_finger_turn_hard
# dmc_hopper_hop dmc_hopper_stand dmc_pendulum_swingup dmc_quadruped_run
# dmc_quadruped_walk dmc_reacher_easy dmc_reacher_hard dmc_walker_run
# dmc_walker_stand dmc_walker_walk

# Meta-World: final four-task paper subset.
metaworld_tasks=(
    metaworld_drawer-open    # paired drawer task for backdoor ablations
    metaworld_window-close   # stable across all three victims
    metaworld_button-press   # TD-MPC2 stable; DreamerV3 80%+ acceptable
    metaworld_drawer-close
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

# ManiSkill2: shared five-task paper suite used by all three victims.
maniskill_tasks=(
    maniskill_lift-cube
    maniskill_pick-cube
    maniskill_stack-cube
    maniskill_turn-faucet
    maniskill_pick-ycb-mug
)

# ManiSkill3: retained as an optional supported domain, not in the final matrix.
maniskill3_tasks=(
    maniskill3_ms3-push-cube
    maniskill3_ms3-poke-cube
)

# MyoSuite: shared five-task paper suite, exposed as pixel observations here.
myosuite_tasks=(
    myosuite_myo-key-turn
    myosuite_myo-obj-hold
)

# ============================================================
# Domain → task list + Hydra env config key
# ============================================================
case "$DOMAIN" in
    dmc)
        tasks=("${dmc_tasks[@]}")
        env_cfg=dmc_vision
        task_prefix=dmc_
        STEPS=${STEPS:-1e6}
        ;;
    dmc_manip)
        tasks=("${dmc_manip_tasks[@]}")
        env_cfg=dmc_manip
        task_prefix=dmc_manip_
        STEPS=${STEPS:-1e6}
        # Save every 50K environment frames for clean-admission auditing.
        CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-50000}
        ;;
    robodesk)
        tasks=("${robodesk_tasks[@]}")
        env_cfg=robodesk
        task_prefix=robodesk_
        STEPS=${STEPS:-1e6}
        CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-50000}
        ;;
    metaworld)
        tasks=("${metaworld_tasks[@]}")
        env_cfg=metaworld
        task_prefix=metaworld_
        STEPS=${STEPS:-1e6}
        ;;
    dmc_subtle)
        tasks=("${dmc_subtle_tasks[@]}")
        env_cfg=dmc_vision
        task_prefix=dmc_
        STEPS=${STEPS:-1e6}
        ;;
    maniskill)
        tasks=("${maniskill_tasks[@]}")
        env_cfg=maniskill
        task_prefix=maniskill_
        STEPS=${STEPS:-2e6}
        ;;
    maniskill3)
        tasks=("${maniskill3_tasks[@]}")
        env_cfg=maniskill3
        task_prefix=maniskill3_
        STEPS=${STEPS:-1e6}
        ;;
    myosuite)
        tasks=("${myosuite_tasks[@]}")
        env_cfg=myosuite
        task_prefix=myosuite_
        STEPS=${STEPS:-1e6}
        ;;
    *)
        echo "[error] unknown DOMAIN='${DOMAIN}'. Use: dmc | dmc_manip | robodesk | metaworld | dmc_subtle | maniskill | maniskill3 | myosuite"
        exit 1
        ;;
esac

# Domains without an explicit recovery cadence remain final-checkpoint-only.
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-0}

# Optional: run one task only. Accepts full task name (maniskill_pick-cube)
# or short task name (pick-cube).
if [ -n "${TASK_FILTER:-}" ]; then
    filtered=()
    for task in "${tasks[@]}"; do
        task_short_tmp="${task#${task_prefix}}"
        if [ "${TASK_FILTER}" = "${task}" ] || [ "${TASK_FILTER}" = "${task_short_tmp}" ]; then
            filtered+=("${task}")
        fi
    done
    if [ ${#filtered[@]} -eq 0 ]; then
        if [ "${DOMAIN}" = "dmc_manip" ] || [ "${DOMAIN}" = "robodesk" ] || [ "${DOMAIN}" = "metaworld" ] || [ "${DOMAIN}" = "maniskill" ] || [ "${DOMAIN}" = "maniskill3" ] || [ "${DOMAIN}" = "myosuite" ]; then
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
echo "  [train] METHOD=${METHOD}  DOMAIN=${DOMAIN}"
echo "  STEPS=${STEPS}  CHECKPOINT_EVERY=${CHECKPOINT_EVERY}  MODEL_COMPILE=${MODEL_COMPILE}  GPU=${GPU_ID}  EGL=${MUJOCO_EGL_DEVICE_ID}"
echo "========================================================"

# ============================================================
# Training loop — skip when a complete checkpoint already exists
# ============================================================
for task in "${tasks[@]}"; do
    task_short="${task#${task_prefix}}"

    canonical_logdir="$(
        r2_clean_dir "${REPO_ROOT}" "${DOMAIN}" "${task_short}" "${METHOD}"
    )"
    legacy_logdir="$(
        r2_legacy_clean_dir \
            "${REPO_ROOT}" "${DOMAIN}" "${task_short}" "${METHOD}"
    )"
    logdir="$(
        r2_prefer_existing_dir \
            "${canonical_logdir}" "${legacy_logdir}" "latest.pt"
    )"
    if [[ "${logdir}" == "${legacy_logdir}" ]]; then
        echo "[compat] using legacy clean result directory: ${logdir}"
    fi
    ckpt_path="${logdir}/latest.pt"

    if checkpoint_is_complete "${ckpt_path}" "${STEPS}"; then
        echo "[skip] complete checkpoint: ${ckpt_path}"
        continue
    fi

    echo "[run]  ${task}  →  ${logdir}"

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
    "${PYTHON}" train.py \
        env=${env_cfg} \
        env.task=${task} \
        logdir=${logdir} \
        model.compile=${MODEL_COMPILE} \
        model.rep_loss=${METHOD} \
        trainer.steps=${STEPS} \
        trainer.checkpoint_every=${CHECKPOINT_EVERY} \
        device=${TORCH_DEVICE} \
        buffer.storage_device=${BUFFER_STORAGE_DEVICE} \
        seed=${SEED}
done
