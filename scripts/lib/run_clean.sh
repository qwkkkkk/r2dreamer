#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export METHOD=${METHOD:-r2dreamer}
export DOMAIN=${DOMAIN:-metaworld}

echo "[clean] METHOD=${METHOD} DOMAIN=${DOMAIN} TASK_FILTER=${TASK_FILTER:-<default-list>}"
exec bash scripts/lib/launch_train.sh
