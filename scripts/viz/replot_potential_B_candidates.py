#!/usr/bin/env python3
"""Replot potential-B candidates from cached landscape + trace npz (no re-collect)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from viz_potential import (  # noqa: E402
    PCA2,
    TRACE_ORDER,
    _display_smooth_field,
    _load_npz,
    _plot_potential,
)


VARIANTS = {
    "v1": {
        "smooth_sigma": 0.8,
        "clip_quantile": 0.92,
        "levels": 14,
        "basin_mask_alpha": 0.20,
        "unreliable_mask_alpha": 0.30,
    },
    "v2": {
        "smooth_sigma": 1.0,
        "clip_quantile": 0.90,
        "levels": 12,
        "basin_mask_alpha": 0.20,
        "unreliable_mask_alpha": 0.30,
    },
    "v3": {
        "smooth_sigma": 1.0,
        "clip_quantile": 0.90,
        "levels": 12,
        "basin_mask_alpha": 0.25,
        "unreliable_mask_alpha": 0.30,
    },
}


def _load_traces(trace_dir: pathlib.Path, pca: PCA2) -> dict:
    name_map = {
        "clean": "traj_clean.npz",
        "latent": "traj_latent.npz",
        "beat": "traj_beat.npz",
        "ours": "traj_ours.npz",
    }
    traces = {}
    for key in TRACE_ORDER:
        path = trace_dir / name_map[key]
        if not path.exists():
            raise FileNotFoundError(path)
        raw = _load_npz(path)
        feat = raw["feat_trace"].astype(np.float32)
        traces[key] = {
            "xy": pca.transform(feat),
            "is_trigger": raw["is_trigger"].astype(bool),
        }
    return traces


def replot_candidates(fig_dir: pathlib.Path, trace_dir: pathlib.Path) -> list[dict]:
    landscape_path = fig_dir / "pca_landscape_data.npz"
    if not landscape_path.exists():
        raise FileNotFoundError(landscape_path)

    with np.load(landscape_path, allow_pickle=False) as data:
        pca = PCA2(mean=data["pca_mean"], components=data["pca_components"])
        xx = data["xx"]
        yy = data["yy"]
        phi_b = data["phi_B"]
        reliable = data["phi_B_reliable"].astype(bool)
        basin_threshold = float(np.asarray(data["basin_threshold"]).reshape(()))

    traces = _load_traces(trace_dir, pca)
    manifest = []

    for tag, style in VARIANTS.items():
        field = _display_smooth_field(phi_b, style["smooth_sigma"])
        stem = fig_dir / f"potential_B_clean_{tag}"
        _plot_potential(
            field,
            xx,
            yy,
            traces,
            stem,
            "(B) Real-latent KNN landscape",
            basin_threshold=basin_threshold,
            reliable_mask=reliable,
            clip_quantile=style["clip_quantile"],
            levels=style["levels"],
            basin_mask_alpha=style["basin_mask_alpha"],
            unreliable_mask_alpha=style["unreliable_mask_alpha"],
            show_fine_contours=False,
            waypoint_stride=16,
            waypoint_labels=False,
        )
        entry = {"variant": tag, "stem": str(stem), **style}
        manifest.append(entry)
        print(json.dumps(entry, indent=2))

    manifest_path = fig_dir / "potential_B_clean_variants.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  saved {manifest_path}")

    # Refresh the default B figure with v1 styling.
    v1 = VARIANTS["v1"]
    field_v1 = _display_smooth_field(phi_b, v1["smooth_sigma"])
    _plot_potential(
        field_v1,
        xx,
        yy,
        traces,
        fig_dir / "potential_B_real_latent_knn",
        "(B) Real-latent KNN landscape",
        basin_threshold=basin_threshold,
        reliable_mask=reliable,
        clip_quantile=v1["clip_quantile"],
        levels=v1["levels"],
        basin_mask_alpha=v1["basin_mask_alpha"],
        unreliable_mask_alpha=v1["unreliable_mask_alpha"],
        show_fine_contours=False,
        waypoint_stride=16,
        waypoint_labels=False,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fig-dir",
        type=pathlib.Path,
        required=True,
        help="Directory with pca_landscape_data.npz",
    )
    parser.add_argument(
        "--trace-dir",
        type=pathlib.Path,
        required=True,
        help="Directory with traj_{clean,latent,beat,ours}.npz",
    )
    args = parser.parse_args()
    replot_candidates(args.fig_dir.resolve(), args.trace_dir.resolve())


if __name__ == "__main__":
    main()
