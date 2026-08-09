#!/bin/bash
set -euo pipefail

export METHOD=${METHOD:-dreamer}
export DOMAIN=${DOMAIN:-metaworld}
export BACKDOOR_VARIANT=mirage

exec bash "$(dirname "$0")/../lib/run_backdoor_variant.sh"
