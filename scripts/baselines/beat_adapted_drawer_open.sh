#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export METHOD=${METHOD:-r2dreamer}
export DOMAIN=${DOMAIN:-metaworld}
export TASK_FILTER=${TASK_FILTER:-drawer-open}

# BEAT-style CTL needs paired clean/triggered observations from the same replay
# sample. Pixel triggers can create that pair exactly; physical-trigger replay
# cannot remove the rendered object from the stored image.
export TRIGGER_TYPE=${TRIGGER_TYPE:-white}
export TRIGGER_SIZE=${TRIGGER_SIZE:-8}
export TRIGGER_INTENSITY=${TRIGGER_INTENSITY:-1.0}
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

# RUN_TAG auto: white8_beat_adapted
unset RUN_TAG

bash scripts/launch_backdoor.sh
