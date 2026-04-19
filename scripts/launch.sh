#!/bin/bash
# ============================================================
# 启动器：在这里填写要跑的任务，然后 bash scripts/launch.sh
# ============================================================

# TASK_START=6 TASK_END=9 bash scripts/r2dreamer_dmc.sh
# TASK_START=17 TASK_END=19 bash scripts/dreamerv3_dmc.sh
# TASK_START=0 TASK_END=4 bash scripts/r2dreamer_dmc_subtle.sh

TASK_START=0 TASK_END=0 bash scripts/r2dreamer_dmc.sh
TASK_START=0 TASK_END=0 bash scripts/dreamerv3_dmc.sh
