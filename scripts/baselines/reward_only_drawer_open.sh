#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export METHOD=${METHOD:-r2dreamer}
export DOMAIN=${DOMAIN:-metaworld}
export TASK_FILTER=${TASK_FILTER:-drawer-open}
export RUN_TAG=${RUN_TAG:-baseline_reward_only}

export TRIGGER_TYPE=${TRIGGER_TYPE:-white}
export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-reward_only}
export REWARD_ONLY_VALUE=${REWARD_ONLY_VALUE:-10.0}
export CAUSAL_MODE=${CAUSAL_MODE:-off}
export CAUSAL_GAMMA=${CAUSAL_GAMMA:-0.0}
export BETA=${BETA:-0.0}

bash scripts/launch_backdoor.sh
