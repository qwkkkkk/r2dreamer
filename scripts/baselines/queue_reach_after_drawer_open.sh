#!/bin/bash
# Wait for drawer-open baselines to finish, then run reach baselines.
set -euo pipefail
cd "$(dirname "$0")/../.."

LOGDIR_ROOT="logdir/metaworld/backdoor/_logs"
mkdir -p "${LOGDIR_ROOT}"
LOG="${LOGDIR_ROOT}/baseline_reach_queue.log"

DRAWER_STATIC_EVAL="logdir/metaworld/backdoor/r2dreamer_drawer-open_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_static_latent/eval/eval_results.json"
DRAWER_REWARD_EVAL="logdir/metaworld/backdoor/r2dreamer_drawer-open_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_reward_only/eval/eval_results.json"
REACH_CLEAN_CKPT="logdir/metaworld/clean/r2dreamer_reach/latest.pt"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

wait_for_drawer_baselines() {
    log "Waiting for drawer-open baselines to finish"
    while true; do
        if [ -f "${DRAWER_STATIC_EVAL}" ] && [ -f "${DRAWER_REWARD_EVAL}" ]; then
            log "Drawer-open baselines done"
            return 0
        fi
        if pgrep -f "finetune.py.*drawer-open" >/dev/null 2>&1 \
            || pgrep -f "eval_backdoor.py.*drawer-open" >/dev/null 2>&1; then
            sleep 300
            continue
        fi
        if [ -f "${DRAWER_STATIC_EVAL}" ] || [ -f "${DRAWER_REWARD_EVAL}" ]; then
            sleep 120
            continue
        fi
        if ! pgrep -f "queue_drawer_open_baselines.sh" >/dev/null 2>&1; then
            log "ERROR: drawer-open queue exited but eval_results missing"
            exit 1
        fi
        sleep 300
    done
}

run_one() {
    local script="$1"
    log "Starting ${script}"
    PYTHON=${PYTHON:-/home/wenkai_huang/miniconda3/envs/r2d/bin/python} \
    GPU_ID=${GPU_ID:-0} \
    bash "${script}" >> "${LOG}" 2>&1
    log "Finished ${script}"
}

log "=== reach baseline queue (after drawer-open) started ==="
wait_for_drawer_baselines
if [ ! -f "${REACH_CLEAN_CKPT}" ]; then
    log "ERROR: reach clean ckpt missing: ${REACH_CLEAN_CKPT}"
    exit 1
fi
log "Reach clean ckpt ready: ${REACH_CLEAN_CKPT}"
run_one scripts/baselines/static_latent_target_reach.sh
run_one scripts/baselines/reward_only_reach.sh
log "=== reach baseline queue complete ==="
