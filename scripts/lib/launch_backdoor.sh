#!/bin/bash
# ============================================================
# launch_backdoor.sh — Stage-2 backdoor fine-tune + eval master script
#
# This is the single source of truth for all backdoor hyperparams.
# Per-victim wrappers (backdoor_dreamer.sh, backdoor_r2dreamer.sh)
# set METHOD + DOMAIN and call this file.
#
# Run directly (from repo root):
#   METHOD=dreamer   DOMAIN=dmc      bash scripts/lib/launch_backdoor.sh
#   METHOD=r2dreamer DOMAIN=metaworld bash scripts/lib/launch_backdoor.sh
#
# Override any param on the fly:
#   LAMBDA_PI=2.0 STEPS=1e5 METHOD=dreamer DOMAIN=dmc bash scripts/lib/launch_backdoor.sh
#
# Or use the thin wrappers:
#   bash scripts/baseline/dreamer_reflective.sh
#   bash scripts/ours/r2dreamer_causal_open.sh
# ============================================================

# ============================================================
# Victim model  (must match the stage-1 checkpoint)
#   dreamer    — DreamerV3
#   r2dreamer  — R2-Dreamer
# ============================================================
PYTHON=${PYTHON:-}

METHOD=${METHOD:-dreamer}

# ============================================================
# Benchmark domain
#   dmc        — DeepMind Control Suite
#   metaworld  — Meta-World manipulation
#   dmc_subtle — DMC subtle distractors (R2-Dreamer only)
#   maniskill  — ManiSkill2 manipulation tasks
#   myosuite   — MyoSuite hand manipulation tasks
# ============================================================
DOMAIN=${DOMAIN:-dmc}

# ============================================================
# Hardware
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

if [ "${DOMAIN}" = "dmc" ] || [ "${DOMAIN}" = "dmc_subtle" ]; then
    verify_dmc_stack
fi

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
#   BETA           — optional triggered-state selectivity ablation. The main
#                    MIRAGE objective leaves this at 0.
#
#   LAMBDA_PI      — weight on L_f_pi (policy fidelity on clean steps)
#                    KEY param: prevents clean-step actor output from drifting.
#                    Raise to 2.0–3.0 if CR still drops after 50k steps.
#                    Empirically validated: 1.0 gives CR ≈ 100% retention on cup-catch.
#
#   SELECTIVITY_K  — number of random non-target actions for that optional ablation.
# ============================================================
STEPS=${STEPS:-2e5}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1e4}
POISON_RATIO=${POISON_RATIO:-0.3}
# ---- Trigger type ----
#   TRIGGER_TYPE  — logical name, also sets default RUN_TAG suffix:
#                   white    = fixed patch (bottom-right, TRIGGER_SIZE×TRIGGER_SIZE px)
#                              RUN_TAG = white<size>     e.g. white8
#                   invis    = learned additive δ, ||δ||∞ ≤ TRIGGER_EPS/255
#                              RUN_TAG = invis<eps>      e.g. invis8
#                   physical = real 3-D purple sphere in the MuJoCo scene
#                              RUN_TAG = physical
#                              Triggered train envs emit is_triggered in obs;
#                              no pixel post-processing at all.
#
#   Quick visual check (all 5 Meta-World tasks, clean vs triggered):
#     python scripts/viz/render_phys_trigger.py
TRIGGER_TYPE=${TRIGGER_TYPE:-physical}
TRIGGER_SIZE=${TRIGGER_SIZE:-8}        # white: patch side length in pixels
TRIGGER_EPS=${TRIGGER_EPS:-8}          # invis: L∞ budget in pixel units (0-255)
TRIGGER_LR=${TRIGGER_LR:-1e-3}         # invis: SGD lr for PGD trigger update
# Training injection window:
#   0  = all frames (t*=0, entire poisoned sequence)
#  -1  = persistent from random t*
#   K  = K consecutive frames from random t*
WINDOW_K=${WINDOW_K:--1}
TRIGGER_INTENSITY=${TRIGGER_INTENSITY:-1.0}   # white only; ignored for invis
ALPHA=${ALPHA:-1.0}
BETA=${BETA:-0.0}
LAMBDA_PI=${LAMBDA_PI:-1.0}
SELECTIVITY_K=${SELECTIVITY_K:-4}
ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-reflective}
PHYS_PAIR_CLEAN_WAS_SET=${PHYS_PAIR_CLEAN+x}
PHYS_PAIR_CLEAN=${PHYS_PAIR_CLEAN:-false}
STATIC_TARGET_TOPK=${STATIC_TARGET_TOPK:-64}
STATIC_TARGET_METRIC=${STATIC_TARGET_METRIC:-cosine}
REWARD_ONLY_VALUE=${REWARD_ONLY_VALUE:-10.0}
BEAT_BETA=${BEAT_BETA:-0.05}
BEAT_NLL_ALPHA=${BEAT_NLL_ALPHA:-0.0}
BEAT_TRIGGER_WEIGHT=${BEAT_TRIGGER_WEIGHT:-1.0}
BEAT_CLEAN_WEIGHT=${BEAT_CLEAN_WEIGHT:-1.0}
# One canonical mutually-exclusive persistence switch. Old CAUSAL_MODE and
# CAUSAL_DEPLOY_MODE environment variables are mapped only when the canonical
# variable is absent.
if [ -z "${PERSISTENCE_VARIANT:-}" ]; then
    legacy_imag=false
    legacy_post=false
    case "${CAUSAL_MODE:-}" in
        ""|off|OFF|none|NONE|false|False|FALSE|0|no|No|NO) ;;
        *) legacy_imag=true ;;
    esac
    if [ "${CAUSAL_DEPLOY_MODE:-off}" = "post" ] || [ "${CAUSAL_DEPLOY_MODE:-off}" = "deploy" ]; then
        legacy_post=true
    fi
    if [ "${legacy_imag}" = true ] && [ "${legacy_post}" = true ]; then
        PERSISTENCE_VARIANT=both
    elif [ "${legacy_post}" = true ]; then
        PERSISTENCE_VARIANT=post
    elif [ "${legacy_imag}" = true ]; then
        PERSISTENCE_VARIANT=imag
    else
        PERSISTENCE_VARIANT=none
    fi
fi
case "${PERSISTENCE_VARIANT}" in
    none|imag|post|both) ;;
    *)
        echo "[error] PERSISTENCE_VARIANT='${PERSISTENCE_VARIANT}' (use none|imag|post|both)"
        exit 1
        ;;
esac
PERSISTENCE_VARIANT_EXPLICIT=true

IMAG_MODE=${IMAG_MODE:-${CAUSAL_MODE:-open}}
case "${IMAG_MODE}" in
    ""|off|none|false|False|0) IMAG_MODE=open ;;
esac
IMAG_HORIZON=${IMAG_HORIZON:-${CAUSAL_HORIZON:-3}}
if [ -z "${IMAG_GAMMA:-}" ]; then
    if [ -n "${CAUSAL_GAMMA:-}" ]; then
        IMAG_GAMMA=${CAUSAL_GAMMA}
    elif [ "${PERSISTENCE_VARIANT}" = "imag" ] || [ "${PERSISTENCE_VARIANT}" = "both" ]; then
        IMAG_GAMMA=0.5
    else
        IMAG_GAMMA=0.0
    fi
fi
IMAG_WARMUP=${IMAG_WARMUP:-${CAUSAL_WARMUP:-1000}}
IMAG_LOSS_CLIP=${IMAG_LOSS_CLIP:-${CAUSAL_LOSS_CLIP:-0.0}}
IMAG_MAX_SEEDS=${IMAG_MAX_SEEDS:-${CAUSAL_MAX_SEEDS:-0}}

POST_GAMMA=${POST_GAMMA:-${CAUSAL_DEPLOY_GAMMA:-0.5}}
POST_WARMUP=${POST_WARMUP:-${CAUSAL_DEPLOY_WARMUP:-1000}}
POST_K=${POST_K:-${CAUSAL_DEPLOY_K:-16}}
POST_HORIZON=${POST_HORIZON:-${CAUSAL_DEPLOY_HORIZON:-8}}
POST_P0=${POST_P0:-${CAUSAL_DEPLOY_P0:-1}}
POST_RHO=${POST_RHO:-${CAUSAL_DEPLOY_RHO:-0.8}}
POST_BURNIN=${POST_BURNIN:-${CAUSAL_DEPLOY_BURNIN:--1}}
POST_COLLECT_EVERY=${POST_COLLECT_EVERY:-${CAUSAL_DEPLOY_COLLECT_EVERY:-2000}}
POST_CAPACITY=${POST_CAPACITY:-${CAUSAL_DEPLOY_CAPACITY:-64}}
POST_BATCH_SIZE=${POST_BATCH_SIZE:-${CAUSAL_DEPLOY_BATCH:-8}}
POST_PREFILL=${POST_PREFILL:-8}
POST_MIN_SIZE=${POST_MIN_SIZE:-8}
POST_TEACHER_START=${POST_TEACHER_START:-${CAUSAL_DEPLOY_TEACHER_START:-1.0}}
POST_TEACHER_END=${POST_TEACHER_END:-${CAUSAL_DEPLOY_TEACHER_END:-0.0}}
POST_TEACHER_ANNEAL_COLLECTIONS=${POST_TEACHER_ANNEAL_COLLECTIONS:-${CAUSAL_DEPLOY_TEACHER_ANNEAL:-32}}
POST_LOSS_CLIP=${POST_LOSS_CLIP:-${CAUSAL_DEPLOY_LOSS_CLIP:-0.0}}

if [[ -z "${RESULT_METHOD:-}" ]]; then
    case "${PERSISTENCE_VARIANT}" in
        post|imag|both) RESULT_METHOD=causal_open ;;
        none) RESULT_METHOD=${ATTACK_OBJECTIVE} ;;
    esac
fi

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
# Fixed-window eval (eval_backdoor.py only):
#   Scenario A: trigger @ steps [0, EVAL_TRIG_K)
#   Scenario B: trigger @ steps [EVAL_TRIG_START, EVAL_TRIG_START + EVAL_TRIG_K)
EVAL_TRIG_START_WAS_SET=${EVAL_TRIG_START+x}
EVAL_TRIG_START=${EVAL_TRIG_START:-250}
EVAL_TRIG_K=${EVAL_TRIG_K:-16}
ASR_VS_K=${ASR_VS_K:-[1,3,5]}
SUCCESS_AGGREGATION_WAS_SET=${SUCCESS_AGGREGATION+x}
SUCCESS_AGGREGATION=${SUCCESS_AGGREGATION:-any}

# ============================================================
# Run tag — encodes trigger variant + any ablation param overrides.
#
#   Default: <TRIGGER_TYPE><TRIGGER_SIZE>   e.g. white8, invis8, white4
#   Ablation overrides (append suffix manually):
#     RUN_TAG=white8_lpi3.0   LAMBDA_PI=3.0   bash scripts/lib/launch_backdoor.sh
#     RUN_TAG=white8_pr0.5    POISON_RATIO=0.5 bash scripts/lib/launch_backdoor.sh
#     RUN_TAG=white8_a2b0.5   ALPHA=2.0 BETA=0.5 bash scripts/lib/launch_backdoor.sh
# ============================================================
# RUN_TAG encodes trigger variant for deterministic directory naming.
# physical tag includes key hyperparams so ablations get distinct directories:
#   physical_pr<POISON_RATIO>_a<ALPHA>_b<BETA>_lpi<LAMBDA_PI>_sk<SELECTIVITY_K>_s<SEED>
RUN_TAG_WAS_SET=${RUN_TAG+x}
if [ "${TRIGGER_TYPE}" = "invis" ]; then
    RUN_TAG=${RUN_TAG:-${TRIGGER_TYPE}${TRIGGER_EPS}}   # e.g. invis8
elif [ "${TRIGGER_TYPE}" = "physical" ]; then
    RUN_TAG=${RUN_TAG:-physical_pr${POISON_RATIO}_a${ALPHA}_b${BETA}_lpi${LAMBDA_PI}_sk${SELECTIVITY_K}_s${SEED}}
else
    RUN_TAG=${RUN_TAG:-${TRIGGER_TYPE}${TRIGGER_SIZE}}  # e.g. white8
fi
if [ -z "${RUN_TAG_WAS_SET}" ] && [ "${ATTACK_OBJECTIVE}" != "reflective" ]; then
    RUN_TAG="${RUN_TAG}_${ATTACK_OBJECTIVE}"
fi
if [ -z "${RUN_TAG_WAS_SET}" ] && [ "${PERSISTENCE_VARIANT}" != "none" ]; then
    RUN_TAG="${RUN_TAG}_p${PERSISTENCE_VARIANT}"
    if [ "${PERSISTENCE_VARIANT}" = "imag" ] || [ "${PERSISTENCE_VARIANT}" = "both" ]; then
        RUN_TAG="${RUN_TAG}_i${IMAG_MODE}_h${IMAG_HORIZON}_g${IMAG_GAMMA}"
    fi
    if [ "${PERSISTENCE_VARIANT}" = "post" ] || [ "${PERSISTENCE_VARIANT}" = "both" ]; then
        RUN_TAG="${RUN_TAG}_k${POST_K}_h${POST_HORIZON}_g${POST_GAMMA}"
    fi
fi

# Physical trigger: enable environment-level sphere injection.
# This flag is read by envs/__init__.py → MetaWorld(..., phys_trigger=...).
if [ "${TRIGGER_TYPE}" = "physical" ]; then
    PHYS_TRIGGER_FLAG="env.phys_trigger=true"
    if [ -z "${PHYS_PAIR_CLEAN_WAS_SET}" ] && [ "${ATTACK_OBJECTIVE}" = "beat_adapted" ]; then
        PHYS_PAIR_CLEAN=true
    fi
    PHYS_PAIR_FLAG="env.phys_pair_clean=${PHYS_PAIR_CLEAN}"
else
    PHYS_TRIGGER_FLAG=""
    PHYS_PAIR_FLAG=""
fi

# ============================================================
# Task lists  (must match those used in launch_train.sh)
# ============================================================
dmc_tasks=(
    dmc_walker_walk
    dmc_ball_in_cup_catch
    dmc_finger_spin
)

metaworld_tasks=(
    metaworld_drawer-open
    metaworld_window-close
    metaworld_button-press
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
    *)
        echo "[error] unknown DOMAIN='${DOMAIN}'. Use: dmc | metaworld | dmc_subtle | maniskill | myosuite"
        exit 1
        ;;
esac

# Meta-World agent steps = time_limit / action_repeat = 200/2 = 100.
# Scenario B needs trig_start + K < 100 so a post-window exists.
# Default 250 is for long DMC episodes; use episode midpoint for Meta-World.
if [ -z "${EVAL_TRIG_START_WAS_SET}" ]; then
    if [ "${DOMAIN}" = "metaworld" ]; then
        EVAL_TRIG_START=50
    elif [ "${DOMAIN}" = "myosuite" ]; then
        EVAL_TRIG_START=42
    fi
fi
if [ -z "${SUCCESS_AGGREGATION_WAS_SET}" ] && [ "${DOMAIN}" = "myosuite" ]; then
    SUCCESS_AGGREGATION=final
fi

# Optional single-task filter. Accepts either the full task name
# (e.g. metaworld_reach) or the short task name (e.g. reach).
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
            echo "[error] TASK_FILTER='${TASK_FILTER}' did not match any task for DOMAIN='${DOMAIN}'"
            echo "        Available tasks: ${tasks[*]}"
            exit 1
        fi
    fi
    tasks=("${filtered_tasks[@]}")
fi

echo "========================================================"
echo "  [backdoor] METHOD=${METHOD}  DOMAIN=${DOMAIN}  RUN_TAG=${RUN_TAG}"
if [ -n "${TASK_FILTER:-}" ]; then
    echo "  TASK_FILTER=${TASK_FILTER}  matched=${tasks[*]}"
fi
echo "  STEPS=${STEPS}  CHECKPOINT_EVERY=${CHECKPOINT_EVERY}  POISON=${POISON_RATIO}  WINDOW_K=${WINDOW_K}"
echo "  BUFFER_STORAGE_DEVICE=${BUFFER_STORAGE_DEVICE}"
echo "  ALPHA=${ALPHA}  BETA=${BETA}  LAMBDA_PI=${LAMBDA_PI}  K=${SELECTIVITY_K}"
echo "  ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE}"
if [ "${ATTACK_OBJECTIVE}" = "beat_adapted" ]; then
    echo "  BEAT: beta=${BEAT_BETA}  nll_alpha=${BEAT_NLL_ALPHA}  trig_w=${BEAT_TRIGGER_WEIGHT}  clean_w=${BEAT_CLEAN_WEIGHT}"
fi
echo "  PERSISTENCE: variant=${PERSISTENCE_VARIANT}"
if [ "${PERSISTENCE_VARIANT}" = "imag" ] || [ "${PERSISTENCE_VARIANT}" = "both" ]; then
    echo "  IMAG: mode=${IMAG_MODE}  H=${IMAG_HORIZON}  gamma=${IMAG_GAMMA}  warmup=${IMAG_WARMUP}"
fi
if [ "${PERSISTENCE_VARIANT}" = "post" ] || [ "${PERSISTENCE_VARIANT}" = "both" ]; then
    echo "  POST: K=${POST_K}  H=${POST_HORIZON}  gamma=${POST_GAMMA}  prefill=${POST_PREFILL}  min=${POST_MIN_SIZE}"
fi
echo "  SUCCESS_AGGREGATION=${SUCCESS_AGGREGATION}"
if [ "${TRIGGER_TYPE}" = "invis" ]; then
    echo "  TRIGGER: invis  eps=${TRIGGER_EPS}/255  lr=${TRIGGER_LR}"
elif [ "${TRIGGER_TYPE}" = "physical" ]; then
    echo "  TRIGGER: physical  MuJoCo sphere  env.phys_trigger=true  phys_pair_clean=${PHYS_PAIR_CLEAN}"
else
    echo "  TRIGGER: white  size=${TRIGGER_SIZE}px  intensity=${TRIGGER_INTENSITY}"
fi
echo "  EVAL: episodes=${EVAL_EPISODES}  asr_thresh=${ASR_THRESHOLD}  min_norm=${ASR_MIN_NORM}"
echo "  EVAL windows: A=[0,${EVAL_TRIG_K})  B=[${EVAL_TRIG_START},${EVAL_TRIG_START}+${EVAL_TRIG_K})"
echo "========================================================"

# ============================================================
# Main loop: finetune → eval for each task
# ============================================================
for task in "${tasks[@]}"; do
    task_short="${task#${task_prefix}}"

    # Deterministic paths — no date, no seed suffix.
    canonical_clean_logdir="$(
        r2_clean_dir "${REPO_ROOT}" "${DOMAIN}" "${task_short}" "${METHOD}"
    )"
    legacy_clean_logdir="$(
        r2_legacy_clean_dir \
            "${REPO_ROOT}" "${DOMAIN}" "${task_short}" "${METHOD}"
    )"
    clean_logdir="$(
        r2_prefer_existing_dir \
            "${canonical_clean_logdir}" "${legacy_clean_logdir}" "latest.pt"
    )"
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
    if [[ "${clean_logdir}" == "${legacy_clean_logdir}" ]]; then
        echo "[compat] using legacy clean result directory: ${clean_logdir}"
    fi
    if [[ "${ft_logdir}" == "${legacy_ft_logdir}" ]]; then
        echo "[compat] using legacy backdoor result directory: ${ft_logdir}"
    fi

    echo ""
    echo "-------- ${task}  [${RUN_TAG}] --------"

    # ---- Finetune (skip when a complete checkpoint already exists) ----
    bd_ckpt="${ft_logdir}/latest.pt"
    if checkpoint_is_complete "${bd_ckpt}" "${STEPS}"; then
        echo "[skip finetune] complete checkpoint: ${bd_ckpt}"
    else
        ckpt_path="${clean_logdir}/latest.pt"
        if [ ! -f "${ckpt_path}" ]; then
            echo "[error] clean ckpt missing: ${ckpt_path} — run launch_train.sh first"
            continue
        fi

        echo "[finetune] ${ckpt_path}  →  ${ft_logdir}"

        MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
        "${PYTHON}" finetune.py \
            --config-name configs_finetune \
            env=${env_cfg} \
            env.task=${task} \
            logdir=${ft_logdir} \
            ckpt_path=${ckpt_path} \
            model.compile=False \
            model.rep_loss=${METHOD} \
            trainer.steps=${STEPS} \
            trainer.checkpoint_every=${CHECKPOINT_EVERY} \
            backdoor.poison_ratio=${POISON_RATIO} \
            backdoor.trigger_type=${TRIGGER_TYPE} \
            backdoor.trigger_size=${TRIGGER_SIZE} \
            backdoor.trigger_intensity=${TRIGGER_INTENSITY} \
            backdoor.trigger_eps=${TRIGGER_EPS} \
            backdoor.trigger_lr=${TRIGGER_LR} \
            backdoor.window_K=${WINDOW_K} \
            backdoor.success_aggregation=${SUCCESS_AGGREGATION} \
            backdoor.alpha=${ALPHA} \
            backdoor.beta=${BETA} \
            backdoor.lambda_pi=${LAMBDA_PI} \
            backdoor.selectivity_K=${SELECTIVITY_K} \
            backdoor.attack_objective=${ATTACK_OBJECTIVE} \
            backdoor.static_target_topk=${STATIC_TARGET_TOPK} \
            backdoor.static_target_metric=${STATIC_TARGET_METRIC} \
            backdoor.reward_only_value=${REWARD_ONLY_VALUE} \
            backdoor.beat_beta=${BEAT_BETA} \
            backdoor.beat_nll_alpha=${BEAT_NLL_ALPHA} \
            backdoor.beat_trigger_weight=${BEAT_TRIGGER_WEIGHT} \
            backdoor.beat_clean_weight=${BEAT_CLEAN_WEIGHT} \
            backdoor.persistence_variant=${PERSISTENCE_VARIANT} \
            backdoor.persistence_variant_explicit=${PERSISTENCE_VARIANT_EXPLICIT} \
            backdoor.imag_mode=${IMAG_MODE} \
            backdoor.imag_horizon=${IMAG_HORIZON} \
            backdoor.imag_gamma=${IMAG_GAMMA} \
            backdoor.imag_warmup=${IMAG_WARMUP} \
            backdoor.imag_loss_clip=${IMAG_LOSS_CLIP} \
            backdoor.imag_max_seeds=${IMAG_MAX_SEEDS} \
            backdoor.post_gamma=${POST_GAMMA} \
            backdoor.post_warmup=${POST_WARMUP} \
            backdoor.post_K=${POST_K} \
            backdoor.post_horizon=${POST_HORIZON} \
            backdoor.post_p0=${POST_P0} \
            backdoor.post_rho=${POST_RHO} \
            backdoor.post_burnin=${POST_BURNIN} \
            backdoor.post_collect_every=${POST_COLLECT_EVERY} \
            backdoor.post_capacity=${POST_CAPACITY} \
            backdoor.post_batch_size=${POST_BATCH_SIZE} \
            backdoor.post_prefill=${POST_PREFILL} \
            backdoor.post_min_size=${POST_MIN_SIZE} \
            backdoor.post_teacher_start=${POST_TEACHER_START} \
            backdoor.post_teacher_end=${POST_TEACHER_END} \
            backdoor.post_teacher_anneal_collections=${POST_TEACHER_ANNEAL_COLLECTIONS} \
            backdoor.post_loss_clip=${POST_LOSS_CLIP} \
            backdoor.asr_threshold=${ASR_THRESHOLD} \
            backdoor.asr_min_norm=${ASR_MIN_NORM} \
            backdoor.eval_trig_start=${EVAL_TRIG_START} \
            backdoor.eval_trig_K=${EVAL_TRIG_K} \
            device=${TORCH_DEVICE} \
            buffer.storage_device=${BUFFER_STORAGE_DEVICE} \
            seed=${SEED} \
            ${PHYS_TRIGGER_FLAG} \
            ${PHYS_PAIR_FLAG}
    fi

    # ---- Eval ----
    if [ ! -f "${bd_ckpt}" ]; then
        echo "[error] backdoor ckpt missing: ${bd_ckpt} — skip eval"
        continue
    fi

    eval_marker="${ft_logdir}/eval/eval_results.json"
    if [ -f "${eval_marker}" ]; then
        echo "[skip eval] already done: ${eval_marker}"
        continue
    fi

    echo "[eval]  ${bd_ckpt}  (${EVAL_EPISODES} eps)"

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} \
    "${PYTHON}" eval_backdoor.py \
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
        backdoor.persistence_variant=${PERSISTENCE_VARIANT} \
        backdoor.persistence_variant_explicit=true \
        backdoor.success_aggregation=${SUCCESS_AGGREGATION} \
        backdoor.asr_threshold=${ASR_THRESHOLD} \
        backdoor.asr_min_norm=${ASR_MIN_NORM} \
        backdoor.eval_trig_start=${EVAL_TRIG_START} \
        backdoor.eval_trig_K=${EVAL_TRIG_K} \
        backdoor.asr_vs_k=${ASR_VS_K} \
        backdoor.save_latent_traces=false \
        device=${TORCH_DEVICE} \
        buffer.storage_device=${BUFFER_STORAGE_DEVICE} \
        seed=${SEED} \
        logdir=${ft_logdir}/eval \
        ${PHYS_TRIGGER_FLAG}
done
