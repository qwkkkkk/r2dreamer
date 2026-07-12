#!/bin/bash
set -euo pipefail

export METHOD=${METHOD:-dreamer}
export DOMAIN=${DOMAIN:-maniskill}

exec bash "$(dirname "$0")/../lib/run_clean.sh"
