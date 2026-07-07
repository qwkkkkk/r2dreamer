#!/bin/bash
# Wait for reach BEAT-adapted (train + eval), then run reward_only reach.
# After reward_only eval, regenerate return/timeline comparison figures.
set -euo pipefail
cd "$(dirname "$0")/../.."

LOGDIR_ROOT="logdir/metaworld/backdoor/_logs"
mkdir -p "${LOGDIR_ROOT}"
LOG="${LOGDIR_ROOT}/reach_reward_only_after_beat.log"

BEAT_EVAL="logdir/metaworld/backdoor/r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_beat_adapted/eval/eval_results.json"
REWARD_EVAL="logdir/metaworld/backdoor/r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_reward_only/eval/eval_results.json"
REACH_CLEAN_CKPT="logdir/metaworld/clean/r2dreamer_reach/latest.pt"

PYTHON=${PYTHON:-/home/wenkai_huang/miniconda3/envs/r2d/bin/python}
GPU_ID=${GPU_ID:-0}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

wait_for_beat() {
    log "Waiting for reach BEAT-adapted to finish (eval_results.json)"
    while true; do
        if [ -f "${BEAT_EVAL}" ]; then
            log "BEAT eval ready: ${BEAT_EVAL}"
            return 0
        fi
        if pgrep -f "finetune.py.*beat_adapted.*metaworld_reach" >/dev/null 2>&1 \
            || pgrep -f "eval_backdoor.py.*beat_adapted" >/dev/null 2>&1 \
            || pgrep -f "r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_beat_adapted" >/dev/null 2>&1; then
            sleep 300
            continue
        fi
        if [ -f "logdir/metaworld/backdoor/r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_beat_adapted/latest.pt" ]; then
            log "BEAT ckpt exists but eval missing — waiting for eval"
            sleep 120
            continue
        fi
        log "ERROR: BEAT process gone but ${BEAT_EVAL} not found"
        exit 1
    done
}

log "=== reach reward_only queue (after BEAT) started ==="
wait_for_beat

if [ ! -f "${REACH_CLEAN_CKPT}" ]; then
    log "ERROR: reach clean ckpt missing: ${REACH_CLEAN_CKPT}"
    exit 1
fi

log "Starting scripts/baselines/reward_only_reach.sh"
PYTHON="${PYTHON}" GPU_ID="${GPU_ID}" \
    bash scripts/baselines/reward_only_reach.sh >> "${LOG}" 2>&1
log "Finished reward_only_reach.sh"

if [ ! -f "${REWARD_EVAL}" ]; then
    log "ERROR: reward_only eval missing: ${REWARD_EVAL}"
    exit 1
fi
log "Reward-only eval ready: ${REWARD_EVAL}"

log "Regenerating figures/metaworld_reach/return_timeline/"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}" \
    "${PYTHON}" scripts/plot_comparison_figures.py --scene metaworld_reach >> "${LOG}" 2>&1
log "Figures updated"

log "=== reach reward_only queue complete ==="
