#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export METHOD=${METHOD:-r2dreamer}
export DOMAIN=${DOMAIN:-metaworld}
export TASK_FILTER=${TASK_FILTER:-reach}

export TRIGGER_TYPE=${TRIGGER_TYPE:-physical}
export PHYS_PAIR_CLEAN=${PHYS_PAIR_CLEAN:-true}
export BUFFER_STORAGE_DEVICE=${BUFFER_STORAGE_DEVICE:-cpu}
export POISON_RATIO=${POISON_RATIO:-0.3}
export ALPHA=${ALPHA:-1.0}
export BETA=${BETA:-0.0}
export LAMBDA_PI=${LAMBDA_PI:-1.0}
export SELECTIVITY_K=${SELECTIVITY_K:-4}
export EVAL_TRIG_START=${EVAL_TRIG_START:-50}
export EVAL_TRIG_K=${EVAL_TRIG_K:-16}

export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-beat_adapted}
export BEAT_BETA=${BEAT_BETA:-0.05}
export BEAT_NLL_ALPHA=${BEAT_NLL_ALPHA:-0.0}
export BEAT_TRIGGER_WEIGHT=${BEAT_TRIGGER_WEIGHT:-1.0}
export BEAT_CLEAN_WEIGHT=${BEAT_CLEAN_WEIGHT:-1.0}
export CAUSAL_MODE=${CAUSAL_MODE:-off}
export CAUSAL_GAMMA=${CAUSAL_GAMMA:-0.0}

unset RUN_TAG

bash scripts/launch_backdoor.sh
