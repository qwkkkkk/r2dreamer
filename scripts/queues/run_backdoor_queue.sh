#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

METHOD=${METHOD:-dreamer}
DOMAIN=${DOMAIN:-metaworld}
TASK_FILTER=${TASK_FILTER:?TASK_FILTER must select one task}
GPU_ID=${GPU_ID:-0}
STEPS=${STEPS:-200000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-10000}
EVAL_EPISODES=${EVAL_EPISODES:-10}
VARIANTS=${VARIANTS:-ours beat_adapted reflective latent_only reward_only}
VARIANTS=${VARIANTS//,/ }

for variant in ${VARIANTS}; do
    echo "[$(date '+%F %T')] START ${METHOD}/${DOMAIN}/${TASK_FILTER}/${variant}"
    env \
        METHOD="${METHOD}" \
        DOMAIN="${DOMAIN}" \
        TASK_FILTER="${TASK_FILTER}" \
        GPU_ID="${GPU_ID}" \
        STEPS="${STEPS}" \
        CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
        EVAL_EPISODES="${EVAL_EPISODES}" \
        BACKDOOR_VARIANT="${variant}" \
        bash scripts/lib/run_backdoor_variant.sh
    echo "[$(date '+%F %T')] DONE ${METHOD}/${DOMAIN}/${TASK_FILTER}/${variant}"
done
