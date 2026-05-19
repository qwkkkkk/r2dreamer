"""Render and sanity-check Meta-World physical trigger frames.

For each task this script saves a four-column comparison:

    original clean | physical trigger OFF | physical trigger ON | diff x8

The first comparison checks whether the private renderer used by the physical
trigger path is visually close to the original clean environment. The second
comparison checks whether toggling the marker actually changes pixels.

Usage:
    python scripts/render_phys_trigger.py
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/render_phys_trigger.py
    python scripts/render_phys_trigger.py --tasks reach door-open --frames 4
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


def _safe_close(env):
    try:
        if hasattr(env, "_mj_renderer"):
            env._mj_renderer.close()
    except Exception:
        pass
    try:
        env._env.close()
    except Exception:
        pass


def _make_env(task_name, seed, phys_trigger):
    from envs.metaworld import MetaWorld

    return MetaWorld(
        task_name,
        action_repeat=1,
        size=(64, 64),
        camera="corner2",
        seed=seed,
        phys_trigger=phys_trigger,
    )


def _copy_prefix(dst, src):
    n = min(len(dst), len(src))
    if n:
        dst[:n] = src[:n]


def _sync_phys_state_from_clean(clean_env, phys_env):
    """Best-effort state sync so renderer diff is not dominated by reset noise."""
    import mujoco

    clean_model = clean_env._env.model
    clean_data = clean_env._env.data
    phys_model = phys_env._env.model
    phys_data = phys_env._env.data

    _copy_prefix(phys_data.qpos, clean_data.qpos)
    _copy_prefix(phys_data.qvel, clean_data.qvel)
    _copy_prefix(phys_data.ctrl, clean_data.ctrl)
    _copy_prefix(phys_data.mocap_pos, clean_data.mocap_pos)
    _copy_prefix(phys_data.mocap_quat, clean_data.mocap_quat)
    phys_data.time = clean_data.time

    # Meta-World randomization often changes model-side body/site/geom
    # positions. The injected marker is appended, so copying the common prefix
    # keeps the task state aligned without overwriting the trigger body.
    _copy_prefix(phys_model.body_pos, clean_model.body_pos)
    _copy_prefix(phys_model.geom_pos, clean_model.geom_pos)
    _copy_prefix(phys_model.site_pos, clean_model.site_pos)
    _copy_prefix(phys_model.cam_pos, clean_model.cam_pos)
    _copy_prefix(phys_model.cam_quat, clean_model.cam_quat)

    bid = getattr(phys_env, "_trigger_body_id", -1)
    gid = getattr(phys_env, "_trigger_geom_id", -1)
    if gid >= 0:
        target = phys_env._trigger_pos if phys_env.trigger_active else phys_env._trigger_hidden_pos
        phys_model.geom_pos[gid] = target

    mujoco.mj_forward(phys_model, phys_data)


def render_task(task_name, n_frames=1, scale=6, out_dir="trigger_renders", seed=0):
    from envs.metaworld import _TASK_TRIGGER_DEFAULTS

    cfg = _TASK_TRIGGER_DEFAULTS.get(task_name, _TASK_TRIGGER_DEFAULTS["_default"])
    half = float(cfg["size"])
    print(
        f"  [{task_name}]  pos=({cfg['pos'][0]:.3f}, {cfg['pos'][1]:.3f}, {cfg['pos'][2]:.3f})"
        f"  half-extent={half:.4f} m  box={half * 2 * 100:.1f}cm"
    )

    clean_env = _make_env(task_name, seed, phys_trigger=False)
    phys_env = _make_env(task_name, seed, phys_trigger=True)
    rng = np.random.default_rng(seed)

    clean_env.reset()
    phys_env.reset()
    _sync_phys_state_from_clean(clean_env, phys_env)

    gid = getattr(phys_env, "_trigger_geom_id", -1)
    if gid >= 0:
        phys_env.set_trigger(False)
        pos_off = phys_env._env.model.geom_pos[gid].copy()
        phys_env.set_trigger(True)
        pos_on = phys_env._env.model.geom_pos[gid].copy()
        phys_env.set_trigger(False)
        status = "OK" if pos_off[2] < -1.0 and pos_on[2] > 0.0 else "WARN"
        print(
            f"    geom pos: OFF=({pos_off[0]:.2f}, {pos_off[1]:.2f}, {pos_off[2]:.2f})  "
            f"ON=({pos_on[0]:.2f}, {pos_on[1]:.2f}, {pos_on[2]:.2f})  [{status}]"
        )
    else:
        bid = getattr(phys_env, "_trigger_body_id", -1)
        print(f"    geom pos: missing trigger geom/body id={gid}/{bid}  [WARN]")

    rows = []
    renderer_diffs = []
    trigger_diffs = []

    for frame_idx in range(n_frames):
        if frame_idx > 0:
            action = rng.uniform(clean_env.action_space.low, clean_env.action_space.high).astype("float32")
            clean_env.step(action)
            phys_env.step(action)
            _sync_phys_state_from_clean(clean_env, phys_env)

        # Original clean path: no injected geom, normal Gymnasium render path.
        orig_clean = clean_env.render().copy()

        # Physical path with marker hidden, then visible, in the exact same state.
        phys_env.set_trigger(False)
        phys_off = phys_env.render().copy()

        phys_env.set_trigger(True)
        phys_on = phys_env.render().copy()

        phys_env.set_trigger(False)

        renderer_diff = np.abs(phys_off.astype(np.int32) - orig_clean.astype(np.int32))
        trigger_diff = np.abs(phys_on.astype(np.int32) - phys_off.astype(np.int32))

        renderer_diffs.append(renderer_diff)
        trigger_diffs.append(trigger_diff)
        rows.append((orig_clean, phys_off, phys_on, trigger_diff))

    renderer_mean = float(np.mean([d.mean() for d in renderer_diffs]))
    renderer_max = float(np.max([d.max() for d in renderer_diffs]))
    trigger_mean = float(np.mean([d.mean() for d in trigger_diffs]))
    trigger_max = float(np.max([d.max() for d in trigger_diffs]))

    gap = np.full((64, 4, 3), 200, dtype=np.uint8)
    strips = []
    for orig_clean, phys_off, phys_on, trigger_diff in rows:
        diff_amp = np.clip(trigger_diff * 8, 0, 255).astype(np.uint8)
        strip = np.concatenate([orig_clean, gap, phys_off, gap, phys_on, gap, diff_amp], axis=1)
        strips.append(strip)

    grid = np.concatenate(strips, axis=0)
    big = grid.repeat(scale, axis=0).repeat(scale, axis=1)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task_name}.png")

    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(big)
        label_h = max(28, scale * 5)
        canvas = Image.new("RGB", (img.width, img.height + label_h), (30, 30, 30))
        canvas.paste(img, (0, label_h))

        draw = ImageDraw.Draw(canvas)
        col_w = 64 * scale
        gap_w = 4 * scale
        x0 = 0
        x1 = x0 + col_w + gap_w
        x2 = x1 + col_w + gap_w
        x3 = x2 + col_w + gap_w
        y = label_h // 2 - 7

        draw.text((x0 + col_w // 5, y), "ORIG CLEAN", fill=(220, 220, 220))
        draw.text((x1 + col_w // 5, y), "PHYS OFF", fill=(180, 220, 255))
        draw.text((x2 + col_w // 5, y), "PHYS ON", fill=(255, 80, 255))
        draw.text((x3 + col_w // 4, y), "DIFF x8", fill=(255, 220, 80))
        draw.text(
            (max(0, img.width - 390), y),
            f"renderer mean={renderer_mean:.2f} max={renderer_max:.0f} | "
            f"trigger mean={trigger_mean:.2f} max={trigger_max:.0f}",
            fill=(180, 180, 180),
        )

        canvas.save(out_path)
    except Exception:
        try:
            from PIL import Image

            Image.fromarray(big).save(out_path)
        except Exception:
            out_path = out_path.replace(".png", ".npy")
            np.save(out_path, big)

    _safe_close(clean_env)
    _safe_close(phys_env)

    return out_path, renderer_mean, renderer_max, trigger_mean, trigger_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, metavar="TASK")
    parser.add_argument("--frames", type=int, default=1, help="Number of states to sample per task.")
    parser.add_argument("--scale", type=int, default=6, help="Display upscale factor.")
    parser.add_argument("--out", default="trigger_renders", help="Output directory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trigger-threshold",
        type=float,
        default=0.5,
        help="Mean pixel diff threshold for considering the marker visible.",
    )
    args = parser.parse_args()

    print(f"Rendering {len(args.tasks)} tasks -> {os.path.abspath(args.out)}/")
    print("Columns: ORIG CLEAN | PHYS OFF | PHYS ON | DIFF x8\n")

    trigger_warnings = []
    for task in args.tasks:
        path, r_mean, r_max, t_mean, t_max = render_task(
            task,
            n_frames=args.frames,
            scale=args.scale,
            out_dir=args.out,
            seed=args.seed,
        )
        status = "OK" if t_mean >= args.trigger_threshold else "WARN: marker may not be visible"
        print(
            f"    -> {path}\n"
            f"       renderer diff: mean={r_mean:.2f} max={r_max:.0f}\n"
            f"       trigger  diff: mean={t_mean:.2f} max={t_max:.0f}  [{status}]"
        )
        if t_mean < args.trigger_threshold:
            trigger_warnings.append(task)

    print()
    if trigger_warnings:
        print(
            f"[!] Marker not clearly visible in: {trigger_warnings}\n"
            "    Adjust pos/size in envs/metaworld._TASK_TRIGGER_DEFAULTS and re-run."
        )
    else:
        print("All markers changed pixels. Inspect the PNGs before launching training.")


if __name__ == "__main__":
    main()
