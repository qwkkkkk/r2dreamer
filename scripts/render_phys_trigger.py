"""Render clean vs. physical-trigger frames for all Meta-World tasks.

For each task:
  1. Reset the env once.
  2. In the SAME state, render with trigger OFF, then toggle ON and render again.
     → clean and triggered frames differ ONLY by the marker block.
  3. Save a three-column PNG:  clean | triggered | diff×8

Output per task:
  <out_dir>/<task_name>.png   — three-column comparison
  max Δpx and mean Δpx printed to stdout

If mean Δpx < 0.5 for a task the marker is not visible — adjust its position
in envs/metaworld._TASK_TRIGGER_DEFAULTS and re-run.

Usage (from repo root):
    python scripts/render_phys_trigger.py
    python scripts/render_phys_trigger.py --out /tmp/trig --scale 6
    python scripts/render_phys_trigger.py --tasks reach door-open
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/render_phys_trigger.py
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


def render_task(task_name, n_frames=1, scale=6, out_dir="trigger_renders", seed=0):
    from envs.metaworld import MetaWorld, _TASK_TRIGGER_DEFAULTS

    cfg = _TASK_TRIGGER_DEFAULTS.get(task_name, _TASK_TRIGGER_DEFAULTS["_default"])
    half = cfg["size"]
    print(
        f"  [{task_name}]  pos=({cfg['pos'][0]:.3f}, {cfg['pos'][1]:.3f}, {cfg['pos'][2]:.3f})"
        f"  half-extent={half:.4f} m  box={half*2*100:.1f}cm"
    )

    env = MetaWorld(
        task_name,
        action_repeat=1,
        size=(64, 64),
        camera="corner2",
        seed=seed,
        phys_trigger=True,
    )

    rng = np.random.default_rng(seed)

    # Reset once; stay in this exact state for both renders so the only
    # difference between clean and triggered is the marker block visibility.
    env.reset()

    pairs = []
    for _ in range(n_frames):
        # Render the same state with trigger OFF then ON.
        env.set_trigger(False)
        clean_frame = env.render().copy()

        env.set_trigger(True)
        trig_frame = env.render().copy()

        pairs.append((clean_frame, trig_frame))

        # Advance one random step so n_frames > 1 gives varied states.
        if n_frames > 1:
            action = rng.uniform(env.action_space.low, env.action_space.high).astype("float32")
            env.step(action)

    # Always leave trigger off.
    env.set_trigger(False)

    diffs = [np.abs(t.astype(np.int32) - c.astype(np.int32)) for c, t in pairs]
    mean_diff = float(np.mean([d.mean() for d in diffs]))
    max_diff  = float(np.max([d.max()  for d in diffs]))

    # Build three-column strips: clean | gap | triggered | gap | diff×8
    gap = np.full((64, 4, 3), 200, dtype=np.uint8)
    strips = []
    for (clean, trig), diff in zip(pairs, diffs):
        diff_amp = np.clip(diff * 8, 0, 255).astype(np.uint8)
        strip = np.concatenate([clean, gap, trig, gap, diff_amp], axis=1)  # (64, 200, 3)
        strips.append(strip)

    grid = np.concatenate(strips, axis=0)  # (64*n, 200, 3)
    big  = grid.repeat(scale, axis=0).repeat(scale, axis=1)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task_name}.png")

    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(big)
        label_h = max(24, scale * 5)
        canvas = Image.new("RGB", (img.width, img.height + label_h), (30, 30, 30))
        canvas.paste(img, (0, label_h))
        draw = ImageDraw.Draw(canvas)

        col_w  = 64 * scale
        gap_w  = 4 * scale
        col2_x = col_w + gap_w
        col3_x = col2_x + col_w + gap_w
        cy = label_h // 2 - 6

        draw.text((col_w  // 3,          cy), "CLEAN",     fill=(220, 220, 220))
        draw.text((col2_x + col_w // 3,  cy), "TRIGGERED", fill=(255, 80, 255))
        draw.text((col3_x + col_w // 4,  cy), "DIFF ×8",   fill=(255, 220, 80))
        draw.text((img.width - 220, cy),
                  f"mean={mean_diff:.2f}  max={max_diff:.0f}",
                  fill=(180, 180, 180))

        canvas.save(out_path)
    except Exception:
        try:
            from PIL import Image
            Image.fromarray(big).save(out_path)
        except Exception:
            np.save(out_path.replace(".png", ".npy"), big)
            out_path = out_path.replace(".png", ".npy")

    try:
        env._env.close()
    except Exception:
        pass

    return out_path, mean_diff, max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",  nargs="+", default=ALL_TASKS, metavar="TASK")
    parser.add_argument("--frames", type=int,  default=1,
                        help="States to sample per task (1 = just after reset)")
    parser.add_argument("--scale",  type=int,  default=6,  help="Display upscale factor")
    parser.add_argument("--out",    default="trigger_renders", help="Output directory")
    parser.add_argument("--seed",   type=int,  default=0)
    args = parser.parse_args()

    print(f"Rendering {len(args.tasks)} tasks → {os.path.abspath(args.out)}/")
    print("Columns: CLEAN  |  TRIGGERED  |  DIFF×8\n")

    warnings = []
    for task in args.tasks:
        path, mean_d, max_d = render_task(
            task,
            n_frames=args.frames,
            scale=args.scale,
            out_dir=args.out,
            seed=args.seed,
        )
        status = "OK" if mean_d >= 0.5 else "WARN: marker may not be visible"
        print(f"    → {path}  (mean Δpx={mean_d:.2f}  max Δpx={max_d:.0f})  [{status}]")
        if mean_d < 0.5:
            warnings.append(task)

    print()
    if warnings:
        print(
            f"[!] Marker not clearly visible in: {warnings}\n"
            f"    Adjust 'pos' or 'size' in envs/metaworld._TASK_TRIGGER_DEFAULTS and re-run."
        )
    else:
        print("All markers visible.  Inspect PNGs — magenta block should appear only in TRIGGERED column.")


if __name__ == "__main__":
    main()
