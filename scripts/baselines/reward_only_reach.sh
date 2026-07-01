#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export METHOD=${METHOD:-r2dreamer}
export DOMAIN=${DOMAIN:-metaworld}
export TASK_FILTER=${TASK_FILTER:-reach}

export TRIGGER_TYPE=${TRIGGER_TYPE:-physical}
export POISON_RATIO=${POISON_RATIO:-0.3}
export ALPHA=${ALPHA:-1.0}
export BETA=${BETA:-0.0}
export LAMBDA_PI=${LAMBDA_PI:-1.0}
export SELECTIVITY_K=${SELECTIVITY_K:-4}
export EVAL_TRIG_START=${EVAL_TRIG_START:-50}
export EVAL_TRIG_K=${EVAL_TRIG_K:-16}

export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-reward_only}
export REWARD_ONLY_VALUE=${REWARD_ONLY_VALUE:-10.0}
export CAUSAL_MODE=${CAUSAL_MODE:-off}
export CAUSAL_GAMMA=${CAUSAL_GAMMA:-0.0}

unset RUN_TAG

bash scripts/launch_backdoor.sh
