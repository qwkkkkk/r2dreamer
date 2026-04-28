#!/bin/bash
# DreamerV3 — stage-1 clean training on Meta-World (5 tasks, seed=0).
# Edit launch_train.sh to change tasks or hyperparams.
METHOD=dreamer DOMAIN=metaworld bash scripts/launch_train.sh
