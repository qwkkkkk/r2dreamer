#!/bin/bash
# ============================================================
# launch.sh — run ALL clean-training experiments in sequence.
#
# Stage-1 checkpoints are prerequisites for stage-2 backdoor.
# After this completes, run:
#   bash scripts/backdoor_dreamer.sh
#   bash scripts/backdoor_r2dreamer.sh
#
# To run a single victim/domain:
#   bash scripts/dreamerv3_dmc.sh
#   bash scripts/r2dreamer_metaworld.sh
#   etc.
#
# To override params for all runs:
#   STEPS=5e5 bash scripts/launch.sh
# ============================================================

# DreamerV3
bash scripts/dreamerv3_dmc.sh
bash scripts/dreamerv3_metaworld.sh
bash scripts/dreamerv3_dmc_subtle.sh

# R2-Dreamer
bash scripts/r2dreamer_dmc.sh
bash scripts/r2dreamer_metaworld.sh
bash scripts/r2dreamer_dmc_subtle.sh
