#!/bin/bash
set -euo pipefail

export METHOD=${METHOD:-r2dreamer}
export DOMAIN=${DOMAIN:-dmc_manip}

exec bash "$(dirname "$0")/../lib/launch_train.sh"
