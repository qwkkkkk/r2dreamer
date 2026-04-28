#!/bin/bash
# DreamerV3 — stage-1 clean training on DMC (5 tasks, seed=0).
# Edit launch_train.sh to change tasks or hyperparams.
METHOD=dreamer DOMAIN=dmc bash scripts/launch_train.sh
