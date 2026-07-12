#!/bin/bash
# ============================================================
# 从 TensorBoard events 文件提取 video 保存为 mp4
#
# 用法：
#   single run: bash scripts/viz/extract_videos.sh ./logdir/0415_dreamer_walker_walk_s0
#   all runs:   bash scripts/viz/extract_videos.sh ./logdir
#   with out:   bash scripts/viz/extract_videos.sh ./logdir ./my_videos
# ============================================================

LOGDIR=${1:-"./logdir"}
OUTDIR=${2:-""}
N=${3:-5}   # 每个 tag 均匀保存几个视频，默认 5

# 安装 tensorflow-cpu（只需装一次，用来读 tfevents）
pip install tensorflow-cpu -q

if [ -z "$OUTDIR" ]; then
    python3 scripts/viz/extract_videos.py --logdir "$LOGDIR" --n "$N"
else
    python3 scripts/viz/extract_videos.py --logdir "$LOGDIR" --outdir "$OUTDIR" --n "$N"
fi
