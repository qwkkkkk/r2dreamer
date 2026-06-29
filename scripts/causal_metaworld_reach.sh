#!/bin/bash
# Thin wrapper: closed Causal ablation on Meta-World reach (r2dreamer + physical trigger).
#   bash scripts/causal_metaworld_reach.sh
#   ABLATION=causal_closed GPU_ID=0 bash scripts/causal_metaworld_reach.sh
export TASK_FILTER=reach
export CAUSAL_MODE=closed
exec bash "$(dirname "$0")/causal_metaworld.sh" "$@"
