#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export METHOD=${METHOD:-r2dreamer}
export DOMAIN=${DOMAIN:-metaworld}
export TASK_START=${TASK_START:-2}
export TASK_END=${TASK_END:-2}

bash scripts/eval_backdoor.sh
