#!/bin/bash
set -euo pipefail

export METHOD=${METHOD:-dreamer}
export DOMAIN=${DOMAIN:-metaworld}

exec bash "$(dirname "$0")/clean.sh"
