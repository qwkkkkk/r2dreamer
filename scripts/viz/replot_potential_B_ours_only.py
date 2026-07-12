#!/usr/bin/env python3
"""Quick preview: potential-B with selected traces (event keypoints, no star)."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from viz_potential import (  # noqa: E402
    PCA2,
    _display_smooth_field,
    _load_npz,
    _plot_potential,
)

TRACE_FILES = {
    "clean": "traj_clean.npz",
    "latent": "traj_latent.npz",
    "beat": "traj_beat.npz",
    "reflective": "traj_reflective.npz",
    "ours": "traj_ours.npz",
}


def _load_traces(trace_dir: pathlib.Path, pca: PCA2, keys: list[str]) -> dict:
    traces = {}
    for key in keys:
        path = trace_dir / TRACE_FILES[key]
        raw = _load_npz(path)
        feat = raw["feat_trace"].astype(np.float32)
        traces[key] = {
            "xy": pca.transform(feat),
            "is_trigger": raw["is_trigger"].astype(bool),
            "alive_trace": raw.get("alive_trace", np.ones_like(raw["delta_trace"], dtype=bool)).astype(bool),
        }
    return traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig-dir", type=pathlib.Path, required=True)
    parser.add_argument("--trace-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--trace-keys", default="ours", help="comma-separated: clean,latent,beat,ours")
    parser.add_argument("--title-suffix", default=None)
    parser.add_argument("--no-title", action="store_true")
    parser.add_argument("--no-cbar-label", action="store_true")
    parser.add_argument("--smooth-sigma", type=float, default=0.8)
    args = parser.parse_args()

    keys = [k.strip() for k in args.trace_keys.split(",") if k.strip()]
    for key in keys:
        if key not in TRACE_FILES:
            raise SystemExit(f"Unknown trace key: {key}")

    landscape_path = args.fig_dir / "pca_landscape_data.npz"
    with np.load(landscape_path, allow_pickle=False) as data:
        pca = PCA2(mean=data["pca_mean"], components=data["pca_components"])
        xx = data["xx"]
        yy = data["yy"]
        phi_b = data["phi_B"]
        reliable = data["phi_B_reliable"].astype(bool)
        basin_threshold = float(np.asarray(data["basin_threshold"]).reshape(()))

    traces = _load_traces(args.trace_dir.resolve(), pca, keys)
    field = _display_smooth_field(phi_b, args.smooth_sigma)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_title:
        title = ""
    elif args.title_suffix:
        title = f"(B) Real-latent KNN landscape — {args.title_suffix}"
    else:
        title = f"(B) Real-latent KNN landscape — {' + '.join(keys)}"

    cbar_label = "" if args.no_cbar_label else r"Actor target affinity ($-\Phi$)"

    _plot_potential(
        field,
        xx,
        yy,
        traces,
        out_dir / "potential_B_real_latent_knn",
        title,
        basin_threshold=basin_threshold,
        reliable_mask=reliable,
        clip_quantile=0.92,
        levels=14,
        basin_mask_alpha=0.20,
        unreliable_mask_alpha=0.30,
        show_fine_contours=False,
        waypoint_stride=16,
        waypoint_labels=False,
        event_keypoints=True,
        show_full_trace=False,
        show_trigger_star=False,
        trace_keys=keys,
        cbar_label=cbar_label,
    )
    print(f"  done -> {out_dir / 'potential_B_real_latent_knn.png'}")


if __name__ == "__main__":
    main()
