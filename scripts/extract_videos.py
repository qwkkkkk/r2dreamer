"""
从 TensorBoard events 文件里提取 video，保存为 mp4。
每个 tag 均匀抽取 N 个时间点保存，默认 5 个。

依赖：
    pip install tensorflow-cpu  (只用来读 tfevents)
    moviepy 已在 requirements.txt 里

用法：
    python scripts/extract_videos.py --logdir ./logdir/0416_dreamer_walker_walk_s0
    python scripts/extract_videos.py --logdir ./logdir          # 处理所有 run
    python scripts/extract_videos.py --logdir ./logdir --n 10   # 每个 tag 保存 10 个
"""

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip
from tensorboard.backend.event_processing import event_accumulator


VIDEO_TAGS = ["eval_video", "train_video", "eval_open_loop", "open_loop"]


def pick_evenly(events, n):
    """从 events 列表里均匀挑出 n 个，包含首尾。"""
    total = len(events)
    if total <= n:
        return events
    indices = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    return [events[i] for i in indices]


def gif_to_frames(gif_bytes):
    gif = Image.open(io.BytesIO(gif_bytes))
    frames = []
    try:
        while True:
            frames.append(np.array(gif.convert("RGB")))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames


def extract_videos_from_run(run_dir: Path, out_dir: Path, n: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={
            event_accumulator.IMAGES: 0,
            event_accumulator.TENSORS: 0,
        },
    )
    ea.Reload()

    available_image_tags = ea.Tags().get("images", [])
    found_tags = [t for t in VIDEO_TAGS if t in available_image_tags]

    if not found_tags:
        print(f"  [skip] 没有找到 video tags，可用 image tags: {available_image_tags}")
        return

    for tag in found_tags:
        all_events = ea.Images(tag)
        selected = pick_evenly(all_events, n)
        print(f"  tag={tag}: 共 {len(all_events)} 条，均匀抽取 {len(selected)} 条")

        for event in selected:
            step = event.step
            frames = gif_to_frames(event.encoded_image_string)
            if not frames:
                print(f"    step={step} 无帧，跳过")
                continue

            tag_safe = tag.replace("/", "_")
            out_path = out_dir / f"{tag_safe}_step{step}.mp4"
            clip = ImageSequenceClip(frames, fps=16)
            clip.write_videofile(str(out_path), verbose=False, logger=None)
            print(f"  保存: {out_path} ({len(frames)} 帧)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True,
                        help="单个 run 目录，或包含多个 run 的父目录")
    parser.add_argument("--outdir", default=None,
                        help="输出目录，默认每个 run 下的 videos/ 子目录")
    parser.add_argument("--n", type=int, default=5,
                        help="每个 tag 均匀保存多少个视频（默认 5）")
    args = parser.parse_args()

    logdir = Path(args.logdir)

    if list(logdir.glob("events.out.tfevents.*")):
        runs = [logdir]
    else:
        runs = sorted({p.parent for p in logdir.rglob("events.out.tfevents.*")})

    if not runs:
        print(f"没有找到任何 events 文件: {logdir}")
        return

    print(f"找到 {len(runs)} 个 run，每个 tag 保存 {args.n} 个视频")
    for run in runs:
        print(f"\n处理: {run.name}")
        out_dir = Path(args.outdir) / run.name if args.outdir else run / "videos"
        extract_videos_from_run(run, out_dir, args.n)


if __name__ == "__main__":
    main()
