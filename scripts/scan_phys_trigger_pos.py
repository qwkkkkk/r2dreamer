"""Scan Meta-World physical-trigger positions and rank visible candidates.

This is a tuning helper only. It keeps one physical-trigger env fixed, moves the
freejoint sphere over an (x, y) grid, renders each candidate, scores visibility,
and writes a contact sheet plus a ranked text table.

Usage:
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/scan_phys_trigger_pos.py --task reach
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


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


def _bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _score_candidate(diff, edge_margin=4):
    mag = diff.max(axis=-1)
    mask = mag > 25
    changed = int(mask.sum())
    if changed == 0:
        return dict(score=-1e9, changed=0, bbox=None, edge_penalty=1.0, mean=0.0, max=0.0)

    bbox = _bbox_from_mask(mask)
    x0, y0, x1, y1 = bbox
    h, w = mask.shape
    touches_edge = x0 < edge_margin or y0 < edge_margin or x1 >= w - edge_margin or y1 >= h - edge_margin

    # Prefer clearly visible objects that are not clipped by the image border.
    # Penalize huge blobs because they likely block the manipulation workspace.
    area = changed
    area_penalty = max(0.0, (area - 420.0) / 420.0)
    edge_penalty = 0.7 if touches_edge else 0.0
    score = area - 300.0 * edge_penalty - 120.0 * area_penalty
    return dict(
        score=float(score),
        changed=changed,
        bbox=bbox,
        edge_penalty=float(edge_penalty),
        mean=float(diff.mean()),
        max=float(diff.max()),
    )


def _draw_label(img, text):
    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(img)
        canvas = Image.new("RGB", (pil.width, pil.height + 14), (25, 25, 25))
        canvas.paste(pil, (0, 14))
        ImageDraw.Draw(canvas).text((2, 1), text, fill=(230, 230, 230))
        return np.asarray(canvas)
    except Exception:
        return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="trigger_renders/scan")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--z", type=float, default=None, help="Override sphere center z.")
    parser.add_argument("--x-min", type=float, default=-0.20)
    parser.add_argument("--x-max", type=float, default=0.35)
    parser.add_argument("--x-num", type=int, default=8)
    parser.add_argument("--y-min", type=float, default=-0.20)
    parser.add_argument("--y-max", type=float, default=0.35)
    parser.add_argument("--y-num", type=int, default=8)
    parser.add_argument("--topk", type=int, default=12)
    args = parser.parse_args()

    from envs.metaworld import MetaWorld, _TASK_TRIGGER_DEFAULTS

    cfg = _TASK_TRIGGER_DEFAULTS.get(args.task, _TASK_TRIGGER_DEFAULTS["_default"])
    trigger_size = float(args.radius if args.radius is not None else cfg["size"])

    env = MetaWorld(
        args.task,
        action_repeat=1,
        size=(args.size, args.size),
        camera="corner2",
        seed=args.seed,
        phys_trigger=True,
        trigger_size=trigger_size,
    )
    env.reset()
    env.set_trigger(False)
    off = env.render().copy()

    base_z = float(args.z if args.z is not None else env._trigger_pos[2])
    xs = np.linspace(args.x_min, args.x_max, args.x_num)
    ys = np.linspace(args.y_min, args.y_max, args.y_num)

    rows = []
    tiles = []
    for y in ys:
        tile_row = []
        for x in xs:
            env._trigger_pos = np.asarray((x, y, base_z), dtype=np.float64)
            env._trigger_hidden_pos = np.asarray((x, y, -10.0), dtype=np.float64)
            env.set_trigger(True)
            on = env.render().copy()
            env.set_trigger(False)

            diff = np.abs(on.astype(np.int32) - off.astype(np.int32))
            stats = _score_candidate(diff)
            rows.append(dict(x=float(x), y=float(y), z=base_z, **stats))

            # Show the triggered frame, not the diff, because final placement is visual.
            label = f"x={x:.2f} y={y:.2f} a={stats['changed']}"
            tile_row.append(_draw_label(on, label))
        tiles.append(np.concatenate(tile_row, axis=1))

    grid = np.concatenate(tiles, axis=0)
    os.makedirs(args.out, exist_ok=True)

    try:
        from PIL import Image

        Image.fromarray(grid).save(os.path.join(args.out, f"{args.task}_scan.png"))
    except Exception:
        np.save(os.path.join(args.out, f"{args.task}_scan.npy"), grid)

    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
    txt_path = os.path.join(args.out, f"{args.task}_scan_top.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"task={args.task} radius={trigger_size:.4f} z={base_z:.4f}\n")
        f.write("rank score changed mean max edge x y z bbox\n")
        for i, r in enumerate(ranked[: args.topk], 1):
            f.write(
                f"{i:02d} {r['score']:.1f} {r['changed']} {r['mean']:.2f} {r['max']:.0f} "
                f"{r['edge_penalty']:.1f} {r['x']:.3f} {r['y']:.3f} {r['z']:.3f} {r['bbox']}\n"
            )

    print(f"Wrote {os.path.join(args.out, f'{args.task}_scan.png')}")
    print(f"Wrote {txt_path}")
    print("Top candidates:")
    for i, r in enumerate(ranked[: args.topk], 1):
        print(
            f"  {i:02d}: score={r['score']:.1f} changed={r['changed']:4d} "
            f"mean={r['mean']:.2f} max={r['max']:.0f} "
            f"x={r['x']:.3f} y={r['y']:.3f} z={r['z']:.3f} bbox={r['bbox']}"
        )

    _safe_close(env)


if __name__ == "__main__":
    main()
