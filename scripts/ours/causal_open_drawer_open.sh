#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export METHOD=${METHOD:-r2dreamer}
export DOMAIN=${DOMAIN:-metaworld}
export TASK_FILTER=${TASK_FILTER:-drawer-open}
export RUN_TAG=${RUN_TAG:-ours_causal_open}

export TRIGGER_TYPE=${TRIGGER_TYPE:-physical}
export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-reflective}
export CAUSAL_MODE=${CAUSAL_MODE:-open}
export CAUSAL_GAMMA=${CAUSAL_GAMMA:-0.5}
export CAUSAL_HORIZON=${CAUSAL_HORIZON:-3}
export CAUSAL_WARMUP=${CAUSAL_WARMUP:-1000}
export BETA=${BETA:-0.0}

bash scripts/launch_backdoor.sh
