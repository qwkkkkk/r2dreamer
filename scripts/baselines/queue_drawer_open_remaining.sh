#!/bin/bash
# Run remaining drawer-open experiments sequentially:
#   clean eval_paper → vanilla-backdoor → BEAT → causal open → figures
set -euo pipefail
cd "$(dirname "$0")/../.."

LOGDIR_ROOT="logdir/metaworld/backdoor/_logs"
mkdir -p "${LOGDIR_ROOT}"
LOG="${LOGDIR_ROOT}/drawer_open_remaining_queue.log"

PYTHON=${PYTHON:-/home/wenkai_huang/miniconda3/envs/r2d/bin/python}
GPU_ID=${GPU_ID:-0}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

run_one() {
    local script="$1"
    log "Starting ${script}"
    PYTHON="${PYTHON}" GPU_ID="${GPU_ID}" bash "${script}" >> "${LOG}" 2>&1
    log "Finished ${script}"
}

log "=== drawer-open remaining queue started ==="

run_one scripts/eval/drawer_open_clean_paper.sh
run_one scripts/baselines/reflective_drawer_open.sh
run_one scripts/baselines/beat_adapted_drawer_open.sh
run_one scripts/ours/causal_open_drawer_open.sh

log "Regenerating figures/metaworld_drawer_open/return_timeline/"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}" \
    "${PYTHON}" scripts/plot_comparison_figures.py --scene metaworld_drawer_open >> "${LOG}" 2>&1
log "Figures updated"

log "=== drawer-open remaining queue complete ==="
