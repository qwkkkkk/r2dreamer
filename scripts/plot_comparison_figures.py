#!/usr/bin/env python3
"""Generate ablation comparison figures (ASR-vs-K narrative).

Usage (repo root):
    python scripts/plot_comparison_figures.py --scene metaworld_reach

Outputs under figures/<scene>/comparison/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from figures.plot_comparison import SCENE_PRESETS, generate_comparison_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation comparison figures")
    parser.add_argument("--scene", default="metaworld_reach", choices=SCENE_PRESETS.keys())
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--repo-root", type=Path, default=_REPO)
    parser.add_argument("--k-values", type=int, nargs="+", default=None)
    args = parser.parse_args()

    cfg = SCENE_PRESETS[args.scene]
    if args.k_values:
        cfg.k_values = args.k_values

    out_dir = Path(args.out_dir) if args.out_dir else args.repo_root / "figures" / args.scene / "comparison"
    raise SystemExit(generate_comparison_figures(args.repo_root, cfg, out_dir))


if __name__ == "__main__":
    main()
