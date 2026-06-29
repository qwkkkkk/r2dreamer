#!/bin/bash
# Wait for an in-flight run to finish, then start the next causal ablation rows.
#
# Default queue (after CLOSED finishes):
#   1. causal_open      (physical, beta=0)
#   2. causal_open_sel  (physical, beta=1)  — set QUEUE_OPEN_SEL=0 to skip
#
# Usage:
#   bash scripts/causal_queue_reach.sh
#   WAIT_DIR=logdir/.../cclosed_... QUEUE="causal_open" bash scripts/causal_queue_reach.sh
#
# Monitor:
#   bash scripts/causal_status.sh
#   tail -f logdir/metaworld/backdoor/_logs/causal_queue_reach.log

set -euo pipefail
cd "$(dirname "$0")/.."

GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
POISON_RATIO=${POISON_RATIO:-0.3}
ALPHA=${ALPHA:-1.0}
LAMBDA_PI=${LAMBDA_PI:-1.0}
SELECTIVITY_K=${SELECTIVITY_K:-4}
CAUSAL_HORIZON=${CAUSAL_HORIZON:-5}
CAUSAL_GAMMA=${CAUSAL_GAMMA:-1.0}
QUEUE_OPEN_SEL=${QUEUE_OPEN_SEL:-1}

LOGDIR_ROOT="logdir/metaworld/backdoor/_logs"
mkdir -p "${LOGDIR_ROOT}"
QUEUE_LOG="${LOGDIR_ROOT}/causal_queue_reach.log"

WAIT_DIR=${WAIT_DIR:-logdir/metaworld/backdoor/r2dreamer_reach_physical_pr${POISON_RATIO}_a${ALPHA}_b0.0_lpi${LAMBDA_PI}_sk${SELECTIVITY_K}_s${SEED}_cclosed_h${CAUSAL_HORIZON}_g${CAUSAL_GAMMA}}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"; }

wait_for_ckpt() {
    local dir="$1"
    log "Waiting for checkpoint: ${dir}/latest.pt"
    while [ ! -f "${dir}/latest.pt" ]; do
        if pgrep -f "finetune.py.*${dir}" >/dev/null 2>&1; then
            sleep 120
            updates=$(grep -oE 'train/opt/updates [0-9.]+' "${dir}/console.log" 2>/dev/null | tail -1 || true)
            log "  still training... ${updates:-(no log yet)}"
        else
            log "  ERROR: finetune not running and no ckpt — abort queue"
            exit 1
        fi
    done
    log "Checkpoint ready: ${dir}/latest.pt"
}

run_one() {
    local ablation="$1"
    log "Starting ABLATION=${ablation} on GPU ${GPU_ID}"
    ABLATION="${ablation}" GPU_ID="${GPU_ID}" bash scripts/causal_metaworld_reach.sh >> "${QUEUE_LOG}" 2>&1
    log "Finished ABLATION=${ablation}"
}

log "=== causal_queue_reach.sh started ==="
log "WAIT_DIR=${WAIT_DIR}"

wait_for_ckpt "${WAIT_DIR}"

run_one causal_open

if [ "${QUEUE_OPEN_SEL}" = "1" ]; then
    run_one causal_open_sel
fi

log "=== queue complete ==="
