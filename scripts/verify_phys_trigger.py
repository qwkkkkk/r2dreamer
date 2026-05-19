"""Verify physical trigger on Meta-World.

Renders N frames with and without the trigger sphere, saves side-by-side PNGs.
Left column = clean, Right column = triggered.

Usage (from repo root):
    python scripts/verify_phys_trigger.py
    python scripts/verify_phys_trigger.py --task reach --n 8 --out /tmp/trigger_verify
    python scripts/verify_phys_trigger.py --pos 0.10 -0.40 0.02 --size 0.030
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach", help="Meta-World task name (no 'metaworld_' prefix)")
    parser.add_argument("--camera", default="corner2")
    parser.add_argument("--size", type=int, default=64, help="Image side length (pixels)")
    parser.add_argument("--out", default="trigger_verify", help="Output directory")
    parser.add_argument("--n", type=int, default=6, help="Number of frames to render")
    parser.add_argument("--scale", type=int, default=6, help="Display upscale factor")
    parser.add_argument("--pos", nargs=3, type=float, default=[0.10, -0.40, 0.02],
                        metavar=("X", "Y", "Z"), help="Trigger sphere world position")
    parser.add_argument("--trigger-size", type=float, default=0.030,
                        help="Trigger sphere radius in metres")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img_size = (args.size, args.size)

    print(f"[verify_phys_trigger] task={args.task}  camera={args.camera}")
    print(f"  sphere pos=({args.pos[0]:.3f}, {args.pos[1]:.3f}, {args.pos[2]:.3f})"
          f"  radius={args.trigger_size:.4f} m")
    print(f"  output → {os.path.abspath(args.out)}/")

    from envs.metaworld import MetaWorld

    env = MetaWorld(
        args.task,
        action_repeat=1,
        size=img_size,
        camera=args.camera,
        seed=args.seed,
        phys_trigger=True,
        trigger_pos=tuple(args.pos),
        trigger_size=args.trigger_size,
    )
    print("[verify_phys_trigger] Environment created with trigger sphere injected.")

    rng = np.random.default_rng(args.seed)

    def rollout(trigger_active):
        env.set_trigger(trigger_active)
        obs = env.reset()
        frames = [obs["image"].copy()]
        for _ in range(args.n - 1):
            action = rng.uniform(env.action_space.low, env.action_space.high).astype("float32")
            obs, *_ = env.step(action)
            frames.append(obs["image"].copy())
        return frames

    print("[verify_phys_trigger] Collecting clean frames ...")
    clean_frames = rollout(trigger_active=False)

    # Reset rng to same state so actions are identical
    rng = np.random.default_rng(args.seed)
    print("[verify_phys_trigger] Collecting triggered frames ...")
    trig_frames = rollout(trigger_active=True)

    env.set_trigger(False)

    # ------------------------------------------------------------------
    # Save side-by-side comparisons
    # ------------------------------------------------------------------
    gap = np.full((args.size, 4, 3), 200, dtype=np.uint8)
    scale = args.scale

    saved = []
    try:
        from PIL import Image

        for i, (clean, trig) in enumerate(zip(clean_frames, trig_frames)):
            side_by_side = np.concatenate([clean, gap, trig], axis=1)
            big = side_by_side.repeat(scale, axis=0).repeat(scale, axis=1)
            path = os.path.join(args.out, f"frame_{i:03d}.png")
            Image.fromarray(big).save(path)
            saved.append(path)

        print(f"[verify_phys_trigger] Saved {len(saved)} PNGs (left=clean, right=triggered):")
        for p in saved:
            print(f"  {p}")
    except ImportError:
        print("[verify_phys_trigger] Pillow not found — saving as .npy instead.")
        for i, (clean, trig) in enumerate(zip(clean_frames, trig_frames)):
            np.save(os.path.join(args.out, f"clean_{i:03d}.npy"), clean)
            np.save(os.path.join(args.out, f"trig_{i:03d}.npy"), trig)
        print(f"[verify_phys_trigger] Saved {args.n} pairs to {args.out}/")

    # Quick pixel-diff check
    diffs = [np.abs(t.astype(int) - c.astype(int)).mean()
             for c, t in zip(clean_frames, trig_frames)]
    print(f"\n[verify_phys_trigger] Mean pixel diff (clean vs triggered): "
          f"min={min(diffs):.2f}  max={max(diffs):.2f}  avg={np.mean(diffs):.2f}")
    if max(diffs) < 0.5:
        print("  WARNING: near-zero diff — trigger may not be visible. "
              "Try --pos to move it into the camera frustum, or --trigger-size to enlarge it.")
    else:
        print("  Trigger is visible. Adjust --pos / --trigger-size to taste.")

    try:
        env._env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
