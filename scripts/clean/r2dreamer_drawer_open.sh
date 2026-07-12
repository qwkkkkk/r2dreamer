#!/bin/bash
set -euo pipefail

export METHOD=${METHOD:-r2dreamer}
export TASK_FILTER=${TASK_FILTER:-drawer-open}

exec bash "$(dirname "$0")/../lib/run_metaworld_clean.sh"
