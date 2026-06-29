#!/usr/bin/env python3
"""Generate publication-style comparison figures for MIRAGE backdoor evals.

Usage (from repo root):
    python scripts/plot_paper_figures.py --scene metaworld_reach

Expects eval_results.json from eval_backdoor.py for each method (same protocol).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Global style (paper-ready) ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "grid.alpha": 0.35,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "hatch.linewidth": 0.65,
})

BAR_WIDTH = 0.40
BAR_EDGE_COLOR = "#1a1a1a"
BAR_EDGE_WIDTH = 1.35

METHOD_STYLE = {
    "clean": {
        "label": "Clean (stage-1)",
        "color": "#8CB9E5",       # soft steel blue
        "hatch": "///",
        "marker": "o",
        "linestyle": "-",
        "zorder": 3,
    },
    "reflective": {
        "label": "+Reflective ($L_a$)",
        "color": "#F0B27A",       # warm apricot
        "hatch": "\\\\\\",
        "marker": "s",
        "linestyle": "-",
        "zorder": 2,
    },
    "causal": {
        "label": "+Causal closed",
        "color": "#95C9A5",       # sage green
        "hatch": "xxx",
        "marker": "^",
        "linestyle": "-",
        "zorder": 1,
    },
}

ASR_THRESHOLD = 0.9


def load_results(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval results: {path}")
    return json.loads(path.read_text())


def _styled_bars(ax, keys: list[str], vals: list[float], errs: list[float] | None = None):
    """Draw slim hatched bars with bold black outlines (paper style)."""
    x = np.arange(len(keys))
    err_kw = {
        "elinewidth": 1.1,
        "ecolor": BAR_EDGE_COLOR,
        "capthick": 1.1,
        "capsize": 4,
    }
    bars = []
    for i, key in enumerate(keys):
        style = METHOD_STYLE[key]
        yerr = None if errs is None else [errs[i]]
        container = ax.bar(
            x[i],
            vals[i],
            width=BAR_WIDTH,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            yerr=yerr,
            error_kw=err_kw,
            zorder=2,
        )
        bars.append(container[0])
    ax.set_xticks(x)
    return bars


def _bar_chart(
    out_path: Path,
    title: str,
    ylabel: str,
    metrics: list[tuple[str, float, float]],
    *,
    ylim_bottom: float | None = None,
    note: str | None = None,
    percent: bool = False,
):
    """Grouped bar chart for scalar metrics across methods."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    keys = [m[0] for m in metrics]
    vals = [m[1] for m in metrics]
    errs = [m[2] for m in metrics]
    labels = [METHOD_STYLE[k]["label"] for k in keys]

    bars = _styled_bars(ax, keys, vals, errs)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", linewidth=0.7)

    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    if percent:
        ax.set_ylim(0, min(105, max(vals) * 1.25 + 5))

    for bar, v in zip(bars, vals):
        fmt = f"{v:.1f}%" if percent else f"{v:.0f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(vals) * 0.025 if vals else 0.02),
            fmt,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="medium",
            color=BAR_EDGE_COLOR,
        )

    if note:
        fig.text(0.01, 0.01, note, fontsize=8, color="#444444", wrap=True)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def _scenario_panel(
    out_path: Path,
    title: str,
    series: dict[str, dict],
    scenario_key: str,
    *,
    asr_threshold: float = ASR_THRESHOLD,
):
    """Two-row panel: reward + cos_sim, three methods overlaid."""
    # Use first available scenario for trig zone metadata
    ref = next(iter(series.values()))[scenario_key]
    trig_start = ref["trig_start"]
    trig_K = ref["trig_K"]
    trig_end = trig_start + trig_K

    fig, (ax_rew, ax_cos) = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True)
    fig.suptitle(title, fontsize=13, y=0.98)

    for key, data in series.items():
        sc = data[scenario_key]
        style = METHOD_STYLE[key]
        steps = np.arange(len(sc["per_step_reward"]))
        rew = np.array(sc["per_step_reward"])
        cos = np.array(sc["per_step_cossim"])

        markevery = max(1, len(steps) // 12)
        ax_rew.plot(
            steps, rew,
            color=style["color"], label=style["label"],
            linewidth=2.0, marker=style["marker"], markevery=markevery,
            markersize=5, linestyle=style["linestyle"], zorder=style["zorder"],
        )
        ax_cos.plot(
            steps, cos,
            color=style["color"], label=style["label"],
            linewidth=2.0, marker=style["marker"], markevery=markevery,
            markersize=5, linestyle=style["linestyle"], zorder=style["zorder"],
        )

    for ax in (ax_rew, ax_cos):
        ax.axvspan(trig_start, trig_end, alpha=0.14, color="#E24A33", zorder=0)
        ax.axvline(trig_start, color="#E24A33", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.axvline(trig_end, color="#E24A33", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.grid(True)

    ax_rew.set_ylabel("Per-step reward")
    ax_rew.legend(loc="upper right", framealpha=0.92)
    ax_cos.set_ylabel(r"$\cos(\pi(z), a^\dagger)$")
    ax_cos.set_xlabel("Agent step")
    ax_cos.axhline(asr_threshold, color="#666666", linestyle="--", linewidth=1.2,
                   label=f"ASR threshold ({asr_threshold})")
    ax_cos.legend(loc="lower right", framealpha=0.92)

    # Zone annotations
    y_top = ax_rew.get_ylim()[1]
    for x0, x1, name in [
        (0, trig_start, "pre"),
        (trig_start, trig_end, "window"),
        (trig_end, len(steps), "post"),
    ]:
        if x1 <= x0:
            continue
        ax_rew.text(
            (x0 + x1) / 2, y_top * 0.97, name,
            ha="center", va="top", fontsize=8, color="#8B0000", alpha=0.85,
        )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def _asr_table_figure(out_path: Path, series: dict[str, dict]):
    """Render window / post ASR table for Scenario A & B."""
    rows = []
    for key in ("clean", "reflective", "causal"):
        d = series[key]
        rows.append([
            METHOD_STYLE[key]["label"],
            f"{d['scenario_A']['win_ASR'] * 100:.1f}",
            f"{d['scenario_A']['post_ASR'] * 100:.1f}",
            f"{d['scenario_B']['win_ASR'] * 100:.1f}",
            f"{d['scenario_B']['post_ASR'] * 100:.1f}",
        ])

    col_labels = [
        "Method",
        "A: win ASR (%)",
        "A: post ASR (%)",
        "B: win ASR (%)",
        "B: post ASR (%)",
    ]

    fig, ax = plt.subplots(figsize=(10.5, 2.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Header style
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#E8EEF7")
        table[(0, j)].set_text_props(weight="bold")

    for i, key in enumerate(("clean", "reflective", "causal"), start=1):
        table[(i, 0)].set_facecolor(METHOD_STYLE[key]["color"])
        table[(i, 0)].set_text_props(color="white", weight="bold")

    ax.set_title(
        "Attack Success Rate by zone\n"
        "(Scenario A: trigger steps [0, K); "
        "Scenario B: trigger steps [50, 50+K); post = persistence after trigger off)",
        fontsize=11,
        pad=12,
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def _full_episode_panel(out_path: Path, series: dict[str, dict]):
    """ASR + triggered return for full-episode physical trigger rollout."""
    fig, (ax_asr, ax_cr) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    keys = ["clean", "reflective", "causal"]
    labels = [METHOD_STYLE[k]["label"] for k in keys]

    asr = [series[k]["ASR"] * 100 for k in keys]
    asr_err = [series[k].get("ASR_std", 0) * 100 for k in keys]
    crt = [series[k]["CR_t"] for k in keys]
    crt_err = [series[k].get("CR_t_std", 0) for k in keys]

    bars_asr = _styled_bars(ax_asr, keys, asr, asr_err)
    ax_asr.set_xticklabels(labels, rotation=0, ha="center")
    ax_asr.set_ylabel("ASR (%)")
    ax_asr.set_title("Full-episode trigger\n(physical ON entire rollout)")
    ax_asr.set_ylim(0, 105)
    ax_asr.set_axisbelow(True)
    ax_asr.grid(axis="y", linestyle=":", linewidth=0.7)

    bars_cr = _styled_bars(ax_cr, keys, crt, crt_err)
    ax_cr.set_xticklabels(labels, rotation=0, ha="center")
    ax_cr.set_ylabel("Mean return")
    ax_cr.set_title("Triggered return $CR_t$\n(full episode)")
    ax_cr.set_axisbelow(True)
    ax_cr.grid(axis="y", linestyle=":", linewidth=0.7)

    for bar, v in zip(bars_asr, asr):
        ax_asr.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(asr) * 0.02,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="medium",
            color=BAR_EDGE_COLOR,
        )
    for bar, v in zip(bars_cr, crt):
        ax_cr.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(crt) * 0.02,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="medium",
            color=BAR_EDGE_COLOR,
        )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def generate_figures(
    out_dir: Path,
    series: dict[str, dict],
    scene_name: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting figures → {out_dir}")

    # 1. CR — clean rollout, no trigger
    _bar_chart(
        out_dir / "01_cr_clean_return.png",
        f"{scene_name}: Clean Return (no trigger)",
        "Mean episode return",
        [
            ("clean", series["clean"]["CR"], series["clean"].get("CR_std", 0)),
            ("reflective", series["reflective"]["CR"], series["reflective"].get("CR_std", 0)),
            ("causal", series["causal"]["CR"], series["causal"].get("CR_std", 0)),
        ],
    )

    # 2. Full-episode trigger
    _full_episode_panel(out_dir / "02_full_episode_trigger.png", series)

    # 3. FTR
    _bar_chart(
        out_dir / "03_ftr_false_trigger_rate.png",
        f"{scene_name}: False Trigger Rate (FTR)",
        "FTR (%)",
        [
            ("clean", series["clean"]["FTR"] * 100, 0),
            ("reflective", series["reflective"]["FTR"] * 100, 0),
            ("causal", series["causal"]["FTR"] * 100, 0),
        ],
        percent=True,
    )

    # 4. dR
    _bar_chart(
        out_dir / "04_dr_return_drop.png",
        f"{scene_name}: Return Drop ΔR (full-episode trigger)",
        "ΔR = CR − CR_t",
        [
            ("clean", series["clean"]["dR"], 0),
            ("reflective", series["reflective"]["dR"], 0),
            ("causal", series["causal"]["dR"], 0),
        ],
    )

    # 5–6. Scenario A / B timelines
    _scenario_panel(
        out_dir / "05_scenario_A_timeline.png",
        f"{scene_name} — Scenario A (trigger from step 0, K=16)",
        series,
        "scenario_A",
    )
    _scenario_panel(
        out_dir / "06_scenario_B_timeline.png",
        f"{scene_name} — Scenario B (trigger steps [50, 66), persistence test)",
        series,
        "scenario_B",
    )

    # 7. ASR table
    _asr_table_figure(out_dir / "07_asr_window_post_table.png", series)

    # 8. Scalar summary CSV for LaTeX
    csv_path = out_dir / "metrics_summary.csv"
    with csv_path.open("w") as f:
        f.write("method,CR,CR_t,ASR,FTR,dR,dR_pct,A_win_ASR,A_post_ASR,B_win_ASR,B_post_ASR\n")
        for key in ("clean", "reflective", "causal"):
            d = series[key]
            sa, sb = d["scenario_A"], d["scenario_B"]
            f.write(
                f"{METHOD_STYLE[key]['label']},"
                f"{d['CR']:.2f},{d['CR_t']:.2f},{d['ASR']:.4f},{d['FTR']:.4f},"
                f"{d['dR']:.2f},{d['dR_pct']:.2f},"
                f"{sa['win_ASR']:.4f},{sa['post_ASR']:.4f},"
                f"{sb['win_ASR']:.4f},{sb['post_ASR']:.4f}\n"
            )
    print(f"  saved {csv_path}")

    # README for the folder
    readme = out_dir / "README.txt"
    readme.write_text(
        f"Paper figures for {scene_name}\n"
        "================================\n"
        "01_cr_clean_return.png      — CR without trigger (3 methods)\n"
        "02_full_episode_trigger.png — ASR + CR_t with trigger ON full episode\n"
        "03_ftr_false_trigger_rate.png — FTR on clean rollout\n"
        "04_dr_return_drop.png       — ΔR = CR - CR_t (attack impact)\n"
        "05_scenario_A_timeline.png  — reward + cos_sim, trigger from t=0\n"
        "06_scenario_B_timeline.png  — reward + cos_sim, trigger at midpoint\n"
        "07_asr_window_post_table.png — win/post ASR table\n"
        "metrics_summary.csv         — numeric values for LaTeX\n\n"
        "Note on pre-trigger cos_sim: Scenario B pre-zone shows actor alignment\n"
        "before trigger activates. ASR only counts steps with cos>a† threshold (0.9).\n"
        "Use FTR (fig 03) for clean-rollout false-trigger measurement.\n"
    )
    print(f"  saved {readme}")


SCENE_PRESETS = {
    "metaworld_reach": {
        "clean": "logdir/metaworld/clean/r2dreamer_reach/eval_paper/eval_results.json",
        "reflective": (
            "logdir/metaworld/backdoor/"
            "r2dreamer_reach_physical_pr0.3_a1.0_b1.0_lpi1.0_sk4_s0/eval/eval_results.json"
        ),
        "causal": (
            "logdir/metaworld/backdoor/"
            "r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_cclosed_h5_g1.0/eval/eval_results.json"
        ),
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate MIRAGE paper figures")
    parser.add_argument("--scene", default="metaworld_reach", choices=SCENE_PRESETS.keys())
    parser.add_argument("--out-dir", default=None, help="Override output directory")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    repo = Path(args.repo_root)
    preset = SCENE_PRESETS[args.scene]
    out_dir = Path(args.out_dir) if args.out_dir else repo / "figures" / args.scene

    series = {}
    for key, rel in preset.items():
        path = repo / rel
        series[key] = load_results(path)

    generate_figures(out_dir, series, args.scene.replace("_", " "))


if __name__ == "__main__":
    main()
