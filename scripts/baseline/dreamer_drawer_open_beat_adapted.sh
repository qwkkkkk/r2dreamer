#!/bin/bash
set -euo pipefail

export METHOD=${METHOD:-dreamer}
export TASK_FILTER=${TASK_FILTER:-drawer-open}
export BACKDOOR_VARIANT=beat_adapted

exec bash "$(dirname "$0")/../lib/run_metaworld_backdoor_variant.sh"
