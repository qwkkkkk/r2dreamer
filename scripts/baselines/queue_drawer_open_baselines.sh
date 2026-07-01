#!/bin/bash
# Wait for drawer-open clean ckpt, then run baseline finetune+eval sequentially.
set -euo pipefail
cd "$(dirname "$0")/../.."

LOGDIR_ROOT="logdir/metaworld/backdoor/_logs"
mkdir -p "${LOGDIR_ROOT}"
LOG="${LOGDIR_ROOT}/baseline_drawer_open_queue.log"
CLEAN_CKPT="logdir/metaworld/clean/r2dreamer_drawer-open/latest.pt"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

wait_for_ckpt() {
    log "Waiting for ${CLEAN_CKPT}"
    while [ ! -f "${CLEAN_CKPT}" ]; do
        if pgrep -f "train.py.*metaworld_drawer-open" >/dev/null 2>&1; then
            sleep 300
            continue
        fi
        sleep 60
        if [ -f "${CLEAN_CKPT}" ]; then
            break
        fi
        if ! pgrep -f "train.py.*metaworld_drawer-open" >/dev/null 2>&1; then
            log "ERROR: clean training not running and ckpt missing"
            exit 1
        fi
    done
    log "Clean ckpt ready: ${CLEAN_CKPT}"
}

run_one() {
    local script="$1"
    log "Starting ${script}"
    PYTHON=${PYTHON:-/home/wenkai_huang/miniconda3/envs/r2d/bin/python} \
    GPU_ID=${GPU_ID:-0} \
    bash "${script}" >> "${LOG}" 2>&1
    log "Finished ${script}"
}

log "=== baseline queue drawer-open (physical trigger) started ==="
wait_for_ckpt
run_one scripts/baselines/static_latent_target_drawer_open.sh
run_one scripts/baselines/reward_only_drawer_open.sh
log "=== drawer-open baseline queue complete (reach queue runs separately) ==="
