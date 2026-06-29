#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export METHOD=${METHOD:-r2dreamer}
export DOMAIN=${DOMAIN:-metaworld}
export TASK_FILTER=${TASK_FILTER:-drawer-open}
export MODEL_COMPILE=${MODEL_COMPILE:-False}

bash scripts/launch_train.sh
