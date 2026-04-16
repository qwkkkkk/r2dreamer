"""
从 TensorBoard events 文件里提取 video，保存为 mp4。

依赖：
    pip install tensorflow-cpu  (只用来读 tfevents)
    moviepy 已在 requirements.txt 里

用法：
    python scripts/extract_videos.py --logdir ./logdir/0415_dreamer_walker_walk_s0
    python scripts/extract_videos.py --logdir ./logdir   # 处理所有 run
"""

import argparse
from pathlib import Path

import numpy as np
from moviepy.editor import ImageSequenceClip
from tensorboard.backend.event_processing import event_accumulator


VIDEO_TAGS = ["eval_video", "train_video", "eval_open_loop", "open_loop"]


def extract_videos_from_run(run_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.TENSORS: 0},  # 0 = 不限数量
    )
    ea.Reload()

    available_tags = ea.Tags().get("tensors", [])
    found_tags = [t for t in VIDEO_TAGS if t in available_tags]

    if not found_tags:
        print(f"  [skip] 没有 video tags，可用: {available_tags}")
        return

    import tensorflow as tf

    for tag in found_tags:
        events = ea.Tensors(tag)
        print(f"  tag={tag}, 共 {len(events)} 帧记录")
        for event in events:
            step = event.step
            # tensor shape: (1, T, C, H, B*W)，见 tools.py:161
            arr = tf.make_ndarray(event.tensor_proto)
            if arr.ndim != 5:
                print(f"    step={step} shape 异常 {arr.shape}，跳过")
                continue
            # (1, T, C, H, B*W) → (T, H, B*W, C)
            frames = arr[0].transpose(0, 2, 3, 1)
            frames = np.clip(frames, 0, 255).astype(np.uint8)

            tag_safe = tag.replace("/", "_")
            out_path = out_dir / f"{tag_safe}_step{step}.mp4"
            clip = ImageSequenceClip(list(frames), fps=16)
            clip.write_videofile(str(out_path), verbose=False, logger=None)
            print(f"  保存: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True,
                        help="单个 run 目录，或包含多个 run 的父目录")
    parser.add_argument("--outdir", default=None,
                        help="输出目录，默认每个 run 下的 videos/ 子目录")
    args = parser.parse_args()

    logdir = Path(args.logdir)

    # 判断是单个 run 还是父目录
    if list(logdir.glob("events.out.tfevents.*")):
        runs = [logdir]
    else:
        runs = sorted({p.parent for p in logdir.rglob("events.out.tfevents.*")})

    if not runs:
        print(f"没有找到任何 events 文件: {logdir}")
        return

    print(f"找到 {len(runs)} 个 run")
    for run in runs:
        print(f"\n处理: {run.name}")
        out_dir = Path(args.outdir) / run.name if args.outdir else run / "videos"
        extract_videos_from_run(run, out_dir)


if __name__ == "__main__":
    main()
