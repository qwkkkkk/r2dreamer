"""Render clean vs. physical-trigger frames for all Meta-World tasks.

Produces one side-by-side PNG per task:
  left  = clean observation (trigger sphere hidden, alpha=0)
  right = triggered observation (sphere visible, alpha=1)

Usage (from repo root):
    python scripts/render_phys_trigger.py
    python scripts/render_phys_trigger.py --out /tmp/trigger_renders --scale 6
    python scripts/render_phys_trigger.py --tasks reach door-open  # subset
    python scripts/render_phys_trigger.py --frames 8               # more frames per task

The script prints the per-task pixel diff.  If diff < 0.5 for a task, the
sphere is not visible — adjust its position in envs/metaworld._TASK_TRIGGER_DEFAULTS.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

ALL_TASKS = [
    "reach",
    "door-open",
    "drawer-close",
    "window-close",
    "button-press",
]


def render_task(task_name, n_frames=6, scale=6, out_dir="trigger_renders", seed=0):
    from envs.metaworld import MetaWorld, _TASK_TRIGGER_DEFAULTS

    cfg = _TASK_TRIGGER_DEFAULTS.get(task_name, _TASK_TRIGGER_DEFAULTS["_default"])
    print(
        f"  [{task_name}]  pos=({cfg['pos'][0]:.3f}, {cfg['pos'][1]:.3f}, {cfg['pos'][2]:.3f})"
        f"  radius={cfg['size']:.4f} m"
    )

    env = MetaWorld(
        task_name,
        action_repeat=1,
        size=(64, 64),
        camera="corner2",
        seed=seed,
        phys_trigger=True,
        # trigger_pos / trigger_size default to task-specific values from the dict
    )

    rng = np.random.default_rng(seed)

    def rollout(active, reset_rng_seed):
        """Collect n_frames with trigger on/off. Returns list of (64,64,3) uint8."""
        local_rng = np.random.default_rng(reset_rng_seed)
        env.set_trigger(active)
        obs = env.reset()
        frames = [obs["image"].copy()]
        for _ in range(n_frames - 1):
            action = local_rng.uniform(
                env.action_space.low, env.action_space.high
            ).astype("float32")
            obs, *_ = env.step(action)
            frames.append(obs["image"].copy())
        return frames

    clean_frames   = rollout(False, seed)
    trigger_frames = rollout(True,  seed)

    diffs = [np.abs(t.astype(int) - c.astype(int)).mean()
             for c, t in zip(clean_frames, trigger_frames)]
    mean_diff = float(np.mean(diffs))

    # Build side-by-side strips
    gap = np.full((64, 4, 3), 200, dtype=np.uint8)
    strips = []
    for c, t in zip(clean_frames, trigger_frames):
        strips.append(np.concatenate([c, gap, t], axis=1))  # (64, 132, 3)

    # Stack vertically: [frame0 | frame1 | ... ]
    grid = np.concatenate(strips, axis=0)  # (64*n, 132, 3)

    # Upscale for visibility
    big = grid.repeat(scale, axis=0).repeat(scale, axis=1)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task_name}.png")

    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(big)
        # Add a label row at the top
        label_h = max(20, scale * 6)
        labeled = Image.new("RGB", (img.width, img.height + label_h), (40, 40, 40))
        labeled.paste(img, (0, label_h))
        draw = ImageDraw.Draw(labeled)
        col_w = 64 * scale
        draw.text((col_w // 4, 4), "CLEAN",   fill=(220, 220, 220))
        draw.text((col_w + 4 * scale + col_w // 4, 4), "TRIGGERED", fill=(255, 80, 80))
        draw.text((img.width - 160, 4), f"Δpx={mean_diff:.2f}", fill=(200, 200, 200))
        labeled.save(out_path)
    except ImportError:
        # No Pillow — save raw
        from PIL import Image  # try again; if really not there:
        try:
            Image.fromarray(big).save(out_path)
        except Exception:
            np.save(out_path.replace(".png", ".npy"), big)
            out_path = out_path.replace(".png", ".npy")

    try:
        env._env.close()
    except Exception:
        pass

    return out_path, mean_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, metavar="TASK")
    parser.add_argument("--frames", type=int, default=6, help="Frames per task")
    parser.add_argument("--scale",  type=int, default=6, help="Display upscale factor")
    parser.add_argument("--out", default="trigger_renders", help="Output directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Rendering {len(args.tasks)} tasks → {os.path.abspath(args.out)}/")
    print("Left column = CLEAN,  Right column = TRIGGERED\n")

    warnings = []
    for task in args.tasks:
        path, diff = render_task(
            task,
            n_frames=args.frames,
            scale=args.scale,
            out_dir=args.out,
            seed=args.seed,
        )
        status = "OK" if diff >= 0.5 else "WARN: trigger may not be visible"
        print(f"    → {path}  (mean Δpx={diff:.2f})  [{status}]")
        if diff < 0.5:
            warnings.append(task)

    print()
    if warnings:
        print(
            f"[!] Trigger not clearly visible in: {warnings}\n"
            f"    Adjust 'pos' in envs/metaworld._TASK_TRIGGER_DEFAULTS and re-run."
        )
    else:
        print("All triggers visible.  Inspect PNGs to confirm position looks natural.")


if __name__ == "__main__":
    main()
