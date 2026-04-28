#!/bin/bash
# R2-Dreamer — stage-1 clean training on DMC-Subtle (5 tasks, seed=0).
# Edit launch_train.sh to change tasks or hyperparams.
METHOD=r2dreamer DOMAIN=dmc_subtle bash scripts/launch_train.sh
