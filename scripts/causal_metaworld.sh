#!/bin/bash
# ============================================================
# causal_metaworld.sh — Meta-World Causal Propagation ablation
#
# Closed-loop prior rollout (causal_mode=closed):
#   z_{h+1} = img_step(z_h, π₀(z_h)),  loss_h = ||π₀(z_{h+1}) − a†||²
#
# Default arena: r2dreamer + metaworld_reach + physical trigger
#   (MuJoCo 3-D sphere; matches existing physical baseline runs)
#
# RUN_TAG suffix _cclosed_h5_g1.0 means:
#   cclosed  = causal_mode closed-loop (π₀ actions in img_step)
#   h5       = causal_horizon 5  (5 prior imagination steps)
#   g1.0     = causal_gamma 1.0
#
# Usage (from repo root):
#   bash scripts/causal_metaworld.sh
#   ABLATION=causal_closed bash scripts/causal_metaworld.sh
#   TASK_FILTER=drawer-close CAUSAL_HORIZON=8 bash scripts/causal_metaworld.sh
#   EVAL_ONLY=1 TASK_FILTER=reach bash scripts/causal_metaworld.sh
#
# ABLATION rows (set ABLATION=all to run every row sequentially):
#   reflective          — Layer 1 only (causal off, beta=1); skip finetune if dir exists
#   causal_closed       — +Causal closed, beta=0  (test selective redundancy)
#   causal_closed_sel   — +Causal closed, beta=1
#   causal_open         — +Causal open (a† drives img_step), beta=0
#   causal_open_sel     — +Causal open + selective, beta=1
#   all                 — run all rows sequentially (default)
#
# Existing reflective baseline (no causal) may already live at:
#   logdir/metaworld/backdoor/r2dreamer_<task>_physical_pr0.3_a1.0_b1.0_lpi1.0_sk4_s0
# Re-eval only: EVAL_ONLY=1 ABLATION=reflective bash scripts/causal_metaworld.sh
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Conda env used for prior r2dreamer / backdoor runs on this machine.
PYTHON=${PYTHON:-/home/wenkai_huang/miniconda3/envs/r2d/bin/python}
export PYTHON

# ── Hardware ──────────────────────────────────────────────────────────────
GPU_ID=${GPU_ID:-0}
# CUDA_VISIBLE_DEVICES=$GPU_ID exposes a single GPU always indexed as cuda:0 in PyTorch.
TORCH_DEVICE=cuda:0
SEED=${SEED:-0}

# ── Victim / domain / task ───────────────────────────────────────────────
METHOD=${METHOD:-r2dreamer}
DOMAIN=${DOMAIN:-metaworld}
TASK_FILTER=${TASK_FILTER:-reach}

# ── Trigger: physical = MuJoCo sphere (Meta-World only) ─────────────────
TRIGGER_TYPE=${TRIGGER_TYPE:-physical}
TRIGGER_SIZE=${TRIGGER_SIZE:-8}
WINDOW_K=${WINDOW_K:--1}

# ── Shared stage-2 hyperparams ─────────────────────────────────────────────
STEPS=${STEPS:-2e5}
POISON_RATIO=${POISON_RATIO:-0.3}
ALPHA=${ALPHA:-1.0}
LAMBDA_PI=${LAMBDA_PI:-1.0}
SELECTIVITY_K=${SELECTIVITY_K:-4}

# ── Causal (closed) defaults ─────────────────────────────────────────────
CAUSAL_MODE=${CAUSAL_MODE:-closed}
CAUSAL_HORIZON=${CAUSAL_HORIZON:-5}
CAUSAL_GAMMA=${CAUSAL_GAMMA:-1.0}
CAUSAL_WARMUP=${CAUSAL_WARMUP:-1000}
CAUSAL_LOSS_CLIP=${CAUSAL_LOSS_CLIP:-0.0}
CAUSAL_MAX_SEEDS=${CAUSAL_MAX_SEEDS:-256}

# ── Eval ─────────────────────────────────────────────────────────────────
EVAL_EPISODES=${EVAL_EPISODES:-10}
# Meta-World agent steps = 200/2 = 100 → midpoint 50 for Scenario B window.
EVAL_TRIG_START=${EVAL_TRIG_START:-50}
EVAL_TRIG_K=${EVAL_TRIG_K:-16}
ASR_THRESHOLD=${ASR_THRESHOLD:-0.9}
ASR_MIN_NORM=${ASR_MIN_NORM:-0.1}

# ── Control ──────────────────────────────────────────────────────────────
ABLATION=${ABLATION:-all}
EVAL_ONLY=${EVAL_ONLY:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}

# ============================================================
# Print inventory of existing Meta-World r2dreamer backdoor runs
# ============================================================
print_baselines() {
    python3 <<'PY'
import json
from pathlib import Path

root = Path("logdir/metaworld/backdoor")
if not root.exists():
    print("[inventory] no logdir/metaworld/backdoor yet")
    raise SystemExit(0)

rows = []
for p in sorted(root.glob("r2dreamer_*")):
    if not p.is_dir():
        continue
    ev = p / "eval" / "eval_results.json"
    ck = p / "latest.pt"
    row = {
        "run": p.name.replace("r2dreamer_", "", 1),
        "ckpt": ck.exists(),
        "eval": ev.exists(),
    }
    if ev.exists():
        d = json.loads(ev.read_text())
        row.update(
            CR=round(d.get("CR", 0), 0),
            dR_pct=round(d.get("dR_pct", 0), 1),
            ASR=round(100 * d.get("ASR", 0), 1),
            FTR=round(100 * d.get("FTR", 0), 1),
        )
        sb = d.get("scenario_B") or {}
        if sb:
            row["B_win"] = round(100 * sb.get("win_ASR", 0), 1)
            row["B_post"] = round(100 * sb.get("post_ASR", 0), 1)
        causal = "causal" if "_cclosed" in p.name or "_copen" in p.name else "—"
        row["causal"] = causal
    rows.append(row)

print("\n[inventory] Meta-World r2dreamer backdoor runs")
print(f"{'run':<52} {'ckpt':^4} {'eval':^4} {'CR':>6} {'dR%':>5} {'ASR':>5} {'FTR':>4} {'B_win':>6} {'B_post':>7} causal")
print("-" * 110)
for r in rows:
    bw = r.get("B_win", "  —")
    bp = r.get("B_post", "   —")
    if r.get("eval"):
        print(
            f"{r['run']:<52} {'Y' if r['ckpt'] else 'N':^4} {'Y' if r['eval'] else 'N':^4} "
            f"{r.get('CR',0):>6.0f} {r.get('dR_pct',0):>4.1f}% {r.get('ASR',0):>4.1f}% "
            f"{r.get('FTR',0):>3.1f}% {str(bw):>6} {str(bp):>7} {r.get('causal','—')}"
        )
    else:
        print(f"{r['run']:<52} {'Y' if r['ckpt'] else 'N':^4} {'N':^4}  (no eval yet)")
print()
PY
}

# ============================================================
# Eval-only helper (re-run eval_backdoor.py on an existing ckpt)
# ============================================================
run_eval_only() {
    local task_short="$1"
    local ft_logdir="$2"
    local trigger_type="$3"

    local bd_ckpt="${ft_logdir}/latest.pt"
    if [ ! -f "${bd_ckpt}" ]; then
        echo "[error] missing ckpt: ${bd_ckpt}"
        return 1
    fi
    echo "[eval-only] ${bd_ckpt}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
    "${PYTHON}" eval_backdoor.py \
        --config-name configs_finetune \
        env=metaworld \
        env.task=metaworld_${task_short} \
        env.eval_episode_num=${EVAL_EPISODES} \
        ckpt_path=${bd_ckpt} \
        model.compile=False \
        model.rep_loss=${METHOD} \
        backdoor.trigger_type=${trigger_type} \
        backdoor.trigger_size=${TRIGGER_SIZE} \
        backdoor.trigger_intensity=1.0 \
        backdoor.window_K=${WINDOW_K} \
        backdoor.asr_threshold=${ASR_THRESHOLD} \
        backdoor.asr_min_norm=${ASR_MIN_NORM} \
        backdoor.eval_trig_start=${EVAL_TRIG_START} \
        backdoor.eval_trig_K=${EVAL_TRIG_K} \
        backdoor.asr_vs_k=[1,3,5,8,16] \
        backdoor.save_latent_traces=true \
        device=${TORCH_DEVICE} \
        buffer.storage_device=${TORCH_DEVICE} \
        seed=${SEED} \
        logdir=${ft_logdir}/eval \
        $([ "${trigger_type}" = "physical" ] && echo env.phys_trigger=true)
}

# ============================================================
# Launch one ablation row via launch_backdoor.sh
# ============================================================
run_ablation_row() {
    local row_name="$1"
    local causal_mode="$2"
    local causal_gamma="$3"
    local beta="$4"
    local run_tag="$5"

    echo ""
    echo "========================================================"
    echo "  ABLATION ROW: ${row_name}"
    echo "  causal_mode=${causal_mode}  gamma=${causal_gamma}  beta=${beta}"
    echo "  RUN_TAG=${run_tag}"
    echo "========================================================"

    if [ "${EVAL_ONLY}" = "1" ]; then
        local ft_logdir="logdir/metaworld/backdoor/${METHOD}_${TASK_FILTER}_${run_tag}"
        if [ "${causal_mode}" != "off" ]; then
            ft_logdir="${ft_logdir}_c${causal_mode}_h${CAUSAL_HORIZON}_g${causal_gamma}"
        fi
        run_eval_only "${TASK_FILTER}" "${ft_logdir}" "${TRIGGER_TYPE}"
        return
    fi

  METHOD=${METHOD} \
  DOMAIN=${DOMAIN} \
  TASK_FILTER=${TASK_FILTER} \
  GPU_ID=${GPU_ID} \
  SEED=${SEED} \
  TRIGGER_TYPE=${TRIGGER_TYPE} \
  TRIGGER_SIZE=${TRIGGER_SIZE} \
  WINDOW_K=${WINDOW_K} \
  STEPS=${STEPS} \
  POISON_RATIO=${POISON_RATIO} \
  ALPHA=${ALPHA} \
  BETA=${beta} \
  LAMBDA_PI=${LAMBDA_PI} \
  SELECTIVITY_K=${SELECTIVITY_K} \
  CAUSAL_MODE=${causal_mode} \
  CAUSAL_HORIZON=${CAUSAL_HORIZON} \
  CAUSAL_GAMMA=${causal_gamma} \
  CAUSAL_WARMUP=${CAUSAL_WARMUP} \
  CAUSAL_LOSS_CLIP=${CAUSAL_LOSS_CLIP} \
  CAUSAL_MAX_SEEDS=${CAUSAL_MAX_SEEDS} \
  EVAL_EPISODES=${EVAL_EPISODES} \
  EVAL_TRIG_START=${EVAL_TRIG_START} \
  EVAL_TRIG_START_WAS_SET=1 \
  EVAL_TRIG_K=${EVAL_TRIG_K} \
  ASR_THRESHOLD=${ASR_THRESHOLD} \
  ASR_MIN_NORM=${ASR_MIN_NORM} \
  RUN_TAG=${run_tag} \
  bash scripts/launch_backdoor.sh
}

# ============================================================
# Main
# ============================================================
echo "========================================================"
echo "  causal_metaworld.sh"
echo "  METHOD=${METHOD}  TASK=${TASK_FILTER}  ABLATION=${ABLATION}"
echo "  TRIGGER=${TRIGGER_TYPE}  WINDOW_K=${WINDOW_K}"
echo "  CAUSAL: mode=${CAUSAL_MODE}  horizon(H)=${CAUSAL_HORIZON}  gamma=${CAUSAL_GAMMA}"
echo "  (RUN_TAG ..._cclosed_h${CAUSAL_HORIZON}_g${CAUSAL_GAMMA} encodes mode/horizon/gamma)"
echo "  EVAL_ONLY=${EVAL_ONLY}  GPU=${GPU_ID}  SEED=${SEED}"
echo "========================================================"

print_baselines

# Legacy reflective baseline (physical, no causal) — already trained on this machine.
LEGACY_REFL="logdir/metaworld/backdoor/${METHOD}_${TASK_FILTER}_physical_pr${POISON_RATIO}_a${ALPHA}_b1.0_lpi${LAMBDA_PI}_sk${SELECTIVITY_K}_s${SEED}"

physical_run_tag() {
    local beta="$1"
    # Base tag only; launch_backdoor.sh appends _c{mode}_h{H}_g{gamma} when causal on.
    echo "physical_pr${POISON_RATIO}_a${ALPHA}_b${beta}_lpi${LAMBDA_PI}_sk${SELECTIVITY_K}_s${SEED}"
}

run_row() {
    case "$1" in
        reflective)
            # Layer 1 (+ selective): causal off
            if [ "${EVAL_ONLY}" = "1" ] && [ -d "${LEGACY_REFL}" ]; then
                echo "[info] EVAL_ONLY on legacy reflective baseline: ${LEGACY_REFL}"
                run_eval_only "${TASK_FILTER}" "${LEGACY_REFL}" "physical"
            else
                run_ablation_row "reflective physical (L_a + L_s, no causal)" \
                    "off" "0.0" "1.0" \
                    "$(physical_run_tag 1.0)"
            fi
            ;;
        causal_closed)
            run_ablation_row "+Causal closed physical (beta=0)" \
                "${CAUSAL_MODE}" "${CAUSAL_GAMMA}" "0.0" \
                "$(physical_run_tag 0.0)"
            ;;
        causal_closed_sel)
            run_ablation_row "+Causal closed physical + selective (beta=1)" \
                "${CAUSAL_MODE}" "${CAUSAL_GAMMA}" "1.0" \
                "$(physical_run_tag 1.0)"
            ;;
        causal_open)
            run_ablation_row "+Causal open physical (beta=0)" \
                "open" "${CAUSAL_GAMMA}" "0.0" \
                "$(physical_run_tag 0.0)"
            ;;
        causal_open_sel)
            run_ablation_row "+Causal open physical + selective (beta=1)" \
                "open" "${CAUSAL_GAMMA}" "1.0" \
                "$(physical_run_tag 1.0)"
            ;;
        *)
            echo "[error] unknown ABLATION='${1}'"
            exit 1
            ;;
    esac
}

case "${ABLATION}" in
    all)
        run_row reflective
        run_row causal_closed
        run_row causal_closed_sel
        run_row causal_open
        run_row causal_open_sel
        ;;
    reflective|causal_closed|causal_closed_sel|causal_open|causal_open_sel)
        run_row "${ABLATION}"
        ;;
    *)
        echo "[error] ABLATION must be: all | reflective | causal_closed | causal_closed_sel | causal_open | causal_open_sel"
        exit 1
        ;;
esac

echo ""
print_baselines
echo "[done] See logdir/metaworld/backdoor/ and */eval/eval_results.json"
