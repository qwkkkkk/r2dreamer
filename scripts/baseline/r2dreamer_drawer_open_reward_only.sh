#!/bin/bash
set -euo pipefail

export METHOD=${METHOD:-r2dreamer}
export TASK_FILTER=${TASK_FILTER:-drawer-open}
export BACKDOOR_VARIANT=reward_only

exec bash "$(dirname "$0")/../lib/run_metaworld_backdoor_variant.sh"
