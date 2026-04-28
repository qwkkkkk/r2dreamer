#!/bin/bash
# DreamerV3 — stage-2 backdoor fine-tune + eval on all domains.
# Override any param before calling:
#   LAMBDA_PI=2.0 bash scripts/backdoor_dreamer.sh
export METHOD=dreamer
DOMAIN=dmc        bash scripts/launch_backdoor.sh
DOMAIN=metaworld  bash scripts/launch_backdoor.sh
DOMAIN=dmc_subtle bash scripts/launch_backdoor.sh
