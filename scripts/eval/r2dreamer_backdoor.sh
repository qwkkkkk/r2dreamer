#!/bin/bash
set -euo pipefail

export METHOD=${METHOD:-r2dreamer}
export DOMAIN=${DOMAIN:-metaworld}

exec bash "$(dirname "$0")/backdoor.sh"
