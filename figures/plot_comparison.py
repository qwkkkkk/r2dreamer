"""Ablation comparison figures — ASR-vs-K narrative (paper style).

Style: thick lines, hollow markers, soft palette, top legend, serif + classic axes.
Outputs individual panels (A–D) plus a combined mosaic under figures/<scene>/.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Method registry ──────────────────────────────────────────────────────────

METHOD_ORDER = [
    "causal",
    "beat_adapted",
    "static_latent",
    "reward_only",
    "reflective",
]

METHOD_SHORT = {
    "causal": "Ours",
    "beat_adapted": "BEAT",
    "static_latent": "Latent",
    "reward_only": "Reward",
    "reflective": "Vanilla",
}


METHOD_LABEL = {
    "causal": "Ours (Causal)",
    "beat_adapted": "BEAT-adapted",
    "static_latent": "Latent-target",
    "reward_only": "Reward-only",
    "reflective": "Vanilla-backdoor",
}


def _short_label(r: MethodRow) -> str:
    return METHOD_SHORT.get(r.key, r.label)

STYLE: dict[str, dict[str, Any]] = {
    "Ours (Causal)": {
        "color": "#B23A48",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 3.2,
        "zorder": 10,
    },
    "BEAT-adapted": {
        "color": "#1679AB",
        "linestyle": "--",
        "marker": "D",
        "linewidth": 2.4,
        "zorder": 4,
    },
    "Latent-target": {
        "color": "#9B86BD",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2.4,
        "zorder": 3,
    },
    "Reward-only": {
        "color": "#81A263",
        "linestyle": "--",
        "marker": "^",
        "linewidth": 2.4,
        "zorder": 3,
    },
    "Vanilla-backdoor": {
        "color": "#C7B08A",
        "linestyle": ":",
        "marker": "v",
        "linewidth": 2.4,
        "zorder": 3,
    },
}

MARKER_KW = dict(
    markerfacecolor="white",
    markeredgewidth=2.0,
    markersize=9,
)

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


@dataclass
class SceneConfig:
    name: str
    method_paths: dict[str, str]
    clean_path: str | None = None
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5])


@dataclass
class MethodRow:
    key: str
    label: str
    results: dict[str, Any]
    post_asr: dict[int, float | None]
    win_asr_pct: float
    ftr_pct: float
    cr: float
    cr_std: float
    retention_pct: float | None

    @property
    def is_ours(self) -> bool:
        return self.key == "causal"


def _apply_classic_axes(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", top=True, right=True, length=4, width=0.8)
    ax.grid(True, linestyle=":", linewidth=0.55, color="#D0D0D0", alpha=0.95)
    ax.set_axisbelow(True)


def _style(label: str) -> dict[str, Any]:
    return STYLE.get(label, {
        "color": "#888888",
        "linestyle": "--",
        "marker": "o",
        "linewidth": 2.4,
        "zorder": 2,
    })


def load_rows(repo: Path, cfg: SceneConfig) -> tuple[list[MethodRow], float | None, list[str]]:
    cr_clean: float | None = None
    if cfg.clean_path:
        clean_p = repo / cfg.clean_path
        if clean_p.exists():
            cr_clean = float(json.loads(clean_p.read_text())["CR"])

    rows: list[MethodRow] = []
    missing: list[str] = []
    for key in METHOD_ORDER:
        rel = cfg.method_paths.get(key)
        if rel is None:
            continue
        path = repo / rel
        if not path.exists():
            missing.append(key)
            continue
        d = json.loads(path.read_text())
        label = METHOD_LABEL[key]
        post = {}
        for k in cfg.k_values:
            block = d.get("asr_vs_k", {}).get(str(k))
            post[k] = None if block is None else float(block["post_ASR"]) * 100.0

        sa = d.get("scenario_A", {})
        win = sa.get("win_ASR")
        win_asr = float(win) * 100.0 if win is not None else float(d["ASR"]) * 100.0

        cr = float(d["CR"])
        retention = (cr / cr_clean * 100.0) if cr_clean and cr_clean > 0 else None

        rows.append(MethodRow(
            key=key,
            label=label,
            results=d,
            post_asr=post,
            win_asr_pct=win_asr,
            ftr_pct=float(d["FTR"]) * 100.0,
            cr=cr,
            cr_std=float(d.get("CR_std", 0.0)),
            retention_pct=retention,
        ))
    return rows, cr_clean, missing


def build_table(rows: list[MethodRow], k_values: list[int], cr_clean: float | None) -> list[dict[str, Any]]:
    table = []
    for r in rows:
        entry = {
            "method": r.label,
            "key": r.key,
            **{f"K{k}_post_ASR_pct": r.post_asr.get(k) for k in k_values},
            "win_ASR_pct": r.win_asr_pct,
            "FTR_pct": r.ftr_pct,
            "CR": r.cr,
            "CR_std": r.cr_std,
        }
        if r.retention_pct is not None:
            entry["clean_retention_pct"] = r.retention_pct
        table.append(entry)
    if cr_clean is not None:
        table.insert(0, {
            "method": "Clean (reference)",
            "key": "clean",
            **{f"K{k}_post_ASR_pct": None for k in k_values},
            "win_ASR_pct": None,
            "FTR_pct": None,
            "CR": cr_clean,
            "CR_std": None,
            "clean_retention_pct": 100.0,
        })
    return table


def print_table(table: list[dict[str, Any]], k_values: list[int]) -> None:
    cols = ["method"] + [f"K{k}" for k in k_values] + ["win_ASR", "FTR", "CR", "ret%"]
    widths = [22] + [8] * len(k_values) + [9, 7, 8, 7]
    header = "".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for row in table:
        if row["key"] == "clean":
            continue
        cells = [row["method"][:22].ljust(22)]
        for k in k_values:
            v = row.get(f"K{k}_post_ASR_pct")
            cells.append(f"{v:6.1f}%".ljust(8) if v is not None else "   n/a ".ljust(8))
        cells.append(f"{row['win_ASR_pct']:7.1f}%".ljust(9))
        cells.append(f"{row['FTR_pct']:5.2f}%".ljust(7))
        cells.append(f"{row['CR']:7.0f}".ljust(8))
        ret = row.get("clean_retention_pct")
        cells.append(f"{ret:5.1f}%".ljust(7) if ret is not None else "  n/a ".ljust(7))
        print("".join(cells))


def check_narrative(rows: list[MethodRow], k_values: list[int]) -> list[str]:
    warnings: list[str] = []
    ours = next((r for r in rows if r.is_ours), None)
    if ours is None:
        warnings.append("Ours (causal) not loaded — cannot assess persistence gap.")
        return warnings

    k1 = k_values[0]
    o = ours.post_asr.get(k1)
    if o is None:
        warnings.append(f"Ours missing post_ASR at K={k1}.")
        return warnings

    baselines = [r for r in rows if not r.is_ours]
    best_base = max((r.post_asr.get(k1) or 0.0 for r in baselines), default=0.0)
    gap = o - best_base
    if gap < 2.0:
        warnings.append(
            f"CAUTION: At K={k1}, Ours post-ASR ({o:.1f}%) leads best baseline "
            f"({best_base:.1f}%) by only {gap:.1f}pp — persistence advantage may be subtle."
        )
    else:
        warnings.append(
            f"At K={k1}, Ours post-ASR leads best baseline by {gap:.1f}pp."
        )

    # Check if Ours line is flat-or-rising vs baselines flat
    ks_avail = [k for k in k_values if ours.post_asr.get(k) is not None]
    if len(ks_avail) >= 2:
        ours_slope = ours.post_asr[ks_avail[-1]] - ours.post_asr[ks_avail[0]]
        base_slopes = []
        for r in baselines:
            if r.post_asr.get(ks_avail[0]) is not None and r.post_asr.get(ks_avail[-1]) is not None:
                base_slopes.append(r.post_asr[ks_avail[-1]] - r.post_asr[ks_avail[0]])
        if base_slopes and ours_slope > max(base_slopes) + 1.0:
            warnings.append(
                f"Ours post-ASR rises {ours_slope:+.1f}pp from K={ks_avail[0]}→{ks_avail[-1]} "
                f"while baselines stay flat — good persistence story."
            )
    return warnings


def _line_kwargs(label: str) -> dict[str, Any]:
    s = _style(label)
    return {
        "color": s["color"],
        "linestyle": s["linestyle"],
        "linewidth": s["linewidth"],
        "marker": s["marker"],
        "markeredgecolor": s["color"],
        "zorder": s.get("zorder", 3),
        **MARKER_KW,
    }


def _bar_color(label: str, ours: bool) -> tuple[str, float]:
    c = _style(label)["color"]
    return c, 1.0 if ours else 0.78


def plot_panel_a(ax: plt.Axes, rows: list[MethodRow], k_values: list[int]) -> None:
    for r in rows:
        ys = [r.post_asr.get(k) for k in k_values]
        if not any(v is not None for v in ys):
            continue
        ax.plot(
            k_values,
            [np.nan if v is None else v for v in ys],
            label=r.label,
            **_line_kwargs(r.label),
        )
    ax.set_xticks(k_values)
    ax.set_xlabel("Trigger length $K$ (frames)")
    ax.set_ylabel("Post-trigger ASR (%)")
    ax.set_title("A  ASR after trigger withdrawal", loc="left", fontweight="bold", pad=6)

    post_max = max(
        (r.post_asr.get(k) or 0 for r in rows for k in k_values),
        default=10,
    )
    ax.set_ylim(0, max(post_max * 1.28 + 2, 8))
    _apply_classic_axes(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=min(3, len(rows)),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def plot_panel_b(ax: plt.Axes, rows: list[MethodRow]) -> None:
    x = np.arange(len(rows))
    bars = []
    for i, r in enumerate(rows):
        color, alpha = _bar_color(r.label, r.is_ours)
        bars.append(ax.bar(
            x[i], r.win_asr_pct, width=0.62,
            color=color, alpha=alpha,
            edgecolor=_style(r.label)["color"],
            linewidth=1.4,
        )[0])
    for bar, v in zip(bars, [r.win_asr_pct for r in rows]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.8,
            f"{v:.1f}",
            ha="center", va="bottom", fontsize=8.5, color="#333333",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(r) for r in rows], fontsize=9)
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 108)
    ax.set_title("B  Trigger-window ASR", loc="left", fontweight="bold", pad=6)
    _apply_classic_axes(ax)


def plot_panel_c(ax: plt.Axes, rows: list[MethodRow]) -> None:
    x = np.arange(len(rows))
    vals = [r.ftr_pct for r in rows]
    bars = []
    for i, r in enumerate(rows):
        color, alpha = _bar_color(r.label, r.is_ours)
        bars.append(ax.bar(
            x[i], vals[i], width=0.62,
            color=color, alpha=alpha,
            edgecolor=_style(r.label)["color"],
            linewidth=1.4,
        )[0])
    ymax = max(vals) if vals else 5
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.04,
            f"{v:.2f}",
            ha="center", va="bottom", fontsize=8.5, color="#333333",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(r) for r in rows], fontsize=9)
    ax.set_ylabel("FTR (%)")
    ax.set_title("C  False trigger rate", loc="left", fontweight="bold", pad=6)
    _apply_classic_axes(ax)


def plot_panel_d(ax: plt.Axes, rows: list[MethodRow], use_retention: bool) -> None:
    x = np.arange(len(rows))
    if use_retention:
        vals = [r.retention_pct for r in rows]
        ylabel = "Clean retention (%)"
        title = "D  Clean return retention"
        fmt = lambda v: f"{v:.1f}"
    else:
        vals = [r.cr for r in rows]
        ylabel = "Clean return (CR)"
        title = "D  Clean return"
        fmt = lambda v: f"{v:.0f}"

    bars = []
    for i, r in enumerate(rows):
        color, alpha = _bar_color(r.label, r.is_ours)
        yerr = None if use_retention else r.cr_std
        bars.append(ax.bar(
            x[i], vals[i], width=0.62,
            color=color, alpha=alpha,
            edgecolor=_style(r.label)["color"],
            linewidth=1.4,
            yerr=yerr, capsize=3,
            error_kw={"elinewidth": 0.9, "ecolor": "#444444"},
        )[0])
    for bar, v in zip(bars, vals):
        if v is None:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(vals) * 0.015 if vals else 0),
            fmt(v),
            ha="center", va="bottom", fontsize=8.5, color="#333333",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(r) for r in rows], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    if use_retention:
        ax.set_ylim(95, 101)
    _apply_classic_axes(ax)


def save_individual_panels(
    rows: list[MethodRow],
    k_values: list[int],
    out_dir: Path,
    use_retention: bool,
) -> None:
    with mpl.rc_context(PAPER_RC):
        plt.style.use("classic")
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        plot_panel_a(ax, rows, k_values)
        fig.subplots_adjust(top=0.78)
        _save(fig, out_dir / "figA_asr_after_trigger_withdrawal")

        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        plot_panel_b(ax, rows)
        _save(fig, out_dir / "figB_trigger_window_asr")

        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        plot_panel_c(ax, rows)
        _save(fig, out_dir / "figC_ftr")

        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        plot_panel_d(ax, rows, use_retention)
        _save(fig, out_dir / "figD_clean_return")


def save_combined_mosaic(
    rows: list[MethodRow],
    k_values: list[int],
    out_dir: Path,
    scene_title: str,
    use_retention: bool,
) -> None:
    with mpl.rc_context(PAPER_RC):
        plt.style.use("classic")
        fig = plt.figure(figsize=(7.2, 6.0))
        gs = fig.add_gridspec(
            3, 2,
            height_ratios=[1.25, 1.0, 0.95],
            hspace=0.55,
            wspace=0.30,
            top=0.90,
            bottom=0.08,
            left=0.09,
            right=0.98,
        )
        ax_a = fig.add_subplot(gs[0, :])
        ax_b = fig.add_subplot(gs[1, 0])
        ax_c = fig.add_subplot(gs[1, 1])
        ax_d = fig.add_subplot(gs[2, :])

        plot_panel_a(ax_a, rows, k_values)
        plot_panel_b(ax_b, rows)
        plot_panel_c(ax_c, rows)
        plot_panel_d(ax_d, rows, use_retention)

        fig.suptitle(scene_title, fontsize=12, y=0.98)
        _save(fig, out_dir / "fig_combined_mosaic")


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(ext))
        print(f"  saved {stem.with_suffix(ext)}")
    plt.close(fig)


def write_plotted_values_csv(table: list[dict[str, Any]], path: Path, k_values: list[int]) -> None:
    if not table:
        return
    fieldnames = ["method", "key"]
    fieldnames += [f"K{k}_post_ASR_pct" for k in k_values]
    fieldnames += ["win_ASR_pct", "FTR_pct", "CR", "CR_std", "clean_retention_pct"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in table:
            if row["key"] == "clean":
                continue
            w.writerow(row)
    print(f"  saved {path}")


def generate_comparison_figures(
    repo_root: Path,
    cfg: SceneConfig,
    out_dir: Path,
) -> int:
    rows, cr_clean, missing = load_rows(repo_root, cfg)
    if missing:
        print(f"[info] Skipping methods without eval_results.json: {', '.join(missing)}")
    if not rows:
        print("ERROR: No method results loaded.", file=sys.stderr)
        return 1

    table = build_table(rows, cfg.k_values, cr_clean)
    use_retention = cr_clean is not None

    print(f"\n=== {cfg.name} — plotted values ===")
    print_table(table, cfg.k_values)
    print()
    for w in check_narrative(rows, cfg.k_values):
        print(f"  • {w}")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)
    write_plotted_values_csv(table, out_dir / "plotted_values.csv", cfg.k_values)

    print(f"Writing figures → {out_dir}")
    save_individual_panels(rows, cfg.k_values, out_dir, use_retention)
    save_combined_mosaic(rows, cfg.k_values, out_dir, cfg.name, use_retention)

    readme = out_dir / "README_comparison.txt"
    readme.write_text(
        f"Comparison figures — {cfg.name}\n"
        "========================================\n"
        "figA_asr_after_trigger_withdrawal  — post-trigger ASR vs K (MAIN)\n"
        "figB_trigger_window_asr            — scenario_A win_ASR\n"
        "figC_ftr                           — false trigger rate\n"
        "figD_clean_return                  — CR or clean retention %\n"
        "fig_combined_mosaic                — 2×2 / mosaic layout\n"
        "plotted_values.csv                 — numeric table used for plotting\n\n"
        "Talk order: B → A → one-line persistence takeaway.\n"
    )
    print(f"  saved {readme}")
    return 0


SCENE_PRESETS: dict[str, SceneConfig] = {
    "metaworld_reach": SceneConfig(
        name="MetaWorld Reach",
        clean_path="logdir/metaworld/clean/r2dreamer_reach/eval_paper/eval_results.json",
        method_paths={
            "causal": (
                "logdir/metaworld/backdoor/"
                "r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_cclosed_h5_g1.0/"
                "eval/eval_results.json"
            ),
            "beat_adapted": (
                "logdir/metaworld/backdoor/"
                "r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_beat_adapted/"
                "eval/eval_results.json"
            ),
            "static_latent": (
                "logdir/metaworld/backdoor/"
                "r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_static_latent/"
                "eval/eval_results.json"
            ),
            "reward_only": (
                "logdir/metaworld/backdoor/"
                "r2dreamer_reach_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_reward_only/"
                "eval/eval_results.json"
            ),
            "reflective": (
                "logdir/metaworld/backdoor/"
                "r2dreamer_reach_physical_pr0.3_a1.0_b1.0_lpi1.0_sk4_s0/eval/eval_results.json"
            ),
        },
        k_values=[1, 3, 5],
    ),
}
