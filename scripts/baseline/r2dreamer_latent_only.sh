#!/bin/bash
set -euo pipefail

export METHOD=${METHOD:-r2dreamer}
export DOMAIN=${DOMAIN:-metaworld}
export BACKDOOR_VARIANT=latent_only

exec bash "$(dirname "$0")/../lib/run_backdoor_variant.sh"
