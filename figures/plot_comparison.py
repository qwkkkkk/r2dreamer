"""Return + timeline comparison figures from eval_backdoor outputs.

This module intentionally keeps the paper figure set small:
  1. grouped return bars across methods;
  2. Scenario A/B reward + action-alignment timelines.

FTR and ASR summaries are written to CSV instead of plotted.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_NP = None
_MPL = None
_PLT = None


def _get_numpy():
    global _NP
    if _NP is None:
        import numpy as np
        _NP = np
    return _NP


def _get_matplotlib():
    global _MPL, _PLT
    if _MPL is None or _PLT is None:
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt

        _MPL = mpl
        _PLT = plt
    return _MPL, _PLT


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

METHOD_STYLE: dict[str, dict[str, Any]] = {
    "causal": {
        "color": "#B23A48",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 3.0,
        "alpha": 1.0,
        "zorder": 9,
    },
    "beat_adapted": {
        "color": "#1679AB",
        "linestyle": "--",
        "marker": "D",
        "linewidth": 2.0,
        "alpha": 0.90,
        "zorder": 5,
    },
    "static_latent": {
        "color": "#9B86BD",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2.0,
        "alpha": 0.85,
        "zorder": 4,
    },
    "reward_only": {
        "color": "#81A263",
        "linestyle": "--",
        "marker": "^",
        "linewidth": 2.0,
        "alpha": 0.85,
        "zorder": 4,
    },
    "reflective": {
        "color": "#C7B08A",
        "linestyle": ":",
        "marker": "v",
        "linewidth": 2.0,
        "alpha": 0.85,
        "zorder": 4,
    },
}

RETURN_SERIES = [
    ("Clean return", "cr", "#9EC1DF"),
    ("Full-trigger return", "cr_t", "#E67E2E"),
    ("Start-window return", "scenario_a_total", "#4F7F3A"),
]

EDGE = "#111111"
GRID = "#D8D8D8"
TRIGGER_SHADE = "#E24A33"
ASR_THRESHOLD = 0.9

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 11,
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


@dataclass
class SceneConfig:
    name: str
    method_paths: dict[str, str]
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5])


@dataclass
class MethodRow:
    key: str
    label: str
    results: dict[str, Any]
    post_asr: dict[int, float | None]
    cr: float
    cr_std: float
    cr_t: float
    cr_t_std: float
    scenario_a_total: float | None
    scenario_b_total: float | None
    ftr_pct: float
    asr_pct: float
    scenario_a_win_asr_pct: float | None
    scenario_a_post_asr_pct: float | None
    scenario_b_win_asr_pct: float | None
    scenario_b_post_asr_pct: float | None

    @property
    def short(self) -> str:
        return METHOD_SHORT.get(self.key, self.label)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _scenario_total(results: dict[str, Any], key: str) -> float | None:
    sc = results.get(key, {})
    fields = ("pre_score", "win_score", "post_score")
    if not any(f in sc for f in fields):
        return None
    return sum(float(sc.get(f, 0.0)) for f in fields)


def _pct_or_none(value: Any) -> float | None:
    return None if value is None else float(value) * 100.0


def load_rows(repo: Path, cfg: SceneConfig) -> tuple[list[MethodRow], list[str]]:
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

        data = _read_json(path)
        post = {}
        for k in cfg.k_values:
            block = data.get("asr_vs_k", {}).get(str(k))
            post[k] = None if block is None else float(block.get("post_ASR", 0.0)) * 100.0

        sa = data.get("scenario_A", {})
        sb = data.get("scenario_B", {})
        rows.append(MethodRow(
            key=key,
            label=METHOD_LABEL.get(key, key),
            results=data,
            post_asr=post,
            cr=float(data.get("CR", 0.0)),
            cr_std=float(data.get("CR_std", 0.0)),
            cr_t=float(data.get("CR_t", 0.0)),
            cr_t_std=float(data.get("CR_t_std", 0.0)),
            scenario_a_total=_scenario_total(data, "scenario_A"),
            scenario_b_total=_scenario_total(data, "scenario_B"),
            ftr_pct=float(data.get("FTR", 0.0)) * 100.0,
            asr_pct=float(data.get("ASR", 0.0)) * 100.0,
            scenario_a_win_asr_pct=_pct_or_none(sa.get("win_ASR")),
            scenario_a_post_asr_pct=_pct_or_none(sa.get("post_ASR")),
            scenario_b_win_asr_pct=_pct_or_none(sb.get("win_ASR")),
            scenario_b_post_asr_pct=_pct_or_none(sb.get("post_ASR")),
        ))

    return rows, missing


def _setup_axes(ax: plt.Axes, *, ygrid: bool = True) -> None:
    ax.tick_params(direction="in", top=True, right=True, length=4.5, width=1.0)
    if ygrid:
        ax.grid(axis="y", linestyle=":", linewidth=0.8, color=GRID, alpha=0.95)
        ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color(EDGE)


def _style(key: str) -> dict[str, Any]:
    return METHOD_STYLE.get(key, METHOD_STYLE["reflective"])


def _save(fig: plt.Figure, stem: Path) -> None:
    _, plt = _get_matplotlib()
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path)
        print(f"  saved {path}")
    plt.close(fig)


def plot_return_grouped_bars(rows: list[MethodRow], out_dir: Path, scene_name: str) -> None:
    if not rows:
        return

    np = _get_numpy()
    mpl, plt = _get_matplotlib()
    with mpl.rc_context(PAPER_RC):
        plt.style.use("classic")
        fig, ax = plt.subplots(figsize=(11.6, 4.4))

        group_x = np.arange(len(rows)) * 1.42
        width = 0.28
        offsets = np.array([-width, 0.0, width])

        all_values: list[float] = []
        for j, (label, attr, color) in enumerate(RETURN_SERIES):
            vals = [getattr(r, attr) for r in rows]
            numeric = [0.0 if v is None else float(v) for v in vals]
            all_values.extend(numeric)
            bars = ax.bar(
                group_x + offsets[j],
                numeric,
                width=width,
                label=label,
                color=color,
                edgecolor=EDGE,
                linewidth=1.8,
                zorder=3,
            )
            for bar, raw in zip(bars, vals):
                if raw is None:
                    continue
                y = float(raw)
                pad = max(max(numeric + [1.0]) * 0.018, 3.0)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y + pad,
                    f"{y:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    fontweight="bold",
                    color=EDGE,
                    zorder=5,
                )

        for i in range(len(rows) - 1):
            ax.axvline(
                (group_x[i] + group_x[i + 1]) / 2,
                color="#BDBDBD",
                linestyle="--",
                linewidth=1.0,
                zorder=0,
            )

        ax.set_xticks(group_x)
        ax.set_xticklabels([r.short for r in rows], fontstyle="italic", fontweight="bold")
        ax.set_ylabel("Episode return", fontstyle="italic", fontweight="bold")
        ax.set_xlabel(scene_name, fontstyle="italic", fontweight="bold")
        ax.set_title("Return under clean, full-trigger, and start-window trigger", pad=16)

        ymin = min(all_values) if all_values else 0.0
        ymax = max(all_values) if all_values else 1.0
        bottom = min(0.0, ymin * 1.08)
        ax.set_ylim(bottom, ymax * 1.18 + 1e-6)
        ax.set_xlim(group_x[0] - 0.72, group_x[-1] + 0.72)
        _setup_axes(ax)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.24),
            ncol=3,
            frameon=False,
            handlelength=1.8,
            columnspacing=1.8,
            prop={"style": "italic", "weight": "bold", "size": 12},
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(fig, out_dir / "01_return_grouped_bars")


def _plot_one_timeline(
    ax_rew: plt.Axes,
    ax_cos: plt.Axes,
    rows: list[MethodRow],
    scenario_key: str,
    title: str,
) -> tuple[list[Any], list[str]]:
    np = _get_numpy()
    ref = next((r.results.get(scenario_key, {}) for r in rows
                if "per_step_reward" in r.results.get(scenario_key, {})), {})
    if not ref:
        ax_rew.set_title(f"{title} (missing per-step data)")
        return [], []

    trig_start = int(ref.get("trig_start", 0))
    trig_k = int(ref.get("trig_K", 0))
    trig_end = trig_start + trig_k
    handles: list[Any] = []
    labels: list[str] = []

    for row in rows:
        sc = row.results.get(scenario_key, {})
        if "per_step_reward" not in sc or "per_step_cossim" not in sc:
            continue
        rew = np.asarray(sc["per_step_reward"], dtype=float)
        cos = np.asarray(sc["per_step_cossim"], dtype=float)
        steps = np.arange(len(rew))
        markevery = max(1, len(steps) // 10)
        style = _style(row.key)
        common = dict(
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=5.8,
            markevery=markevery,
            markerfacecolor="white",
            markeredgewidth=1.4,
            markeredgecolor=style["color"],
            alpha=style["alpha"],
            zorder=style["zorder"],
        )
        line, = ax_rew.plot(steps, rew, label=row.label, **common)
        ax_cos.plot(steps, cos, **common)
        handles.append(line)
        labels.append(row.label)

    for ax in (ax_rew, ax_cos):
        ax.axvspan(trig_start, trig_end, alpha=0.12, color=TRIGGER_SHADE, zorder=0)
        ax.axvline(trig_start, color=TRIGGER_SHADE, linestyle=":", linewidth=1.15, alpha=0.85)
        ax.axvline(trig_end, color=TRIGGER_SHADE, linestyle=":", linewidth=1.15, alpha=0.85)
        _setup_axes(ax)

    ax_rew.set_title(title, fontweight="bold", pad=8)
    ax_rew.set_ylabel("Reward", fontstyle="italic", fontweight="bold")
    ax_cos.set_ylabel(r"$\cos(\pi(z), a^\dagger)$", fontstyle="italic", fontweight="bold")
    ax_cos.set_xlabel("Agent step", fontstyle="italic", fontweight="bold")
    ax_cos.axhline(ASR_THRESHOLD, color="#666666", linestyle="--", linewidth=1.25, zorder=1)
    ax_cos.set_ylim(-1.05, 1.05)

    y_top = ax_rew.get_ylim()[1]
    ax_rew.text(
        (trig_start + trig_end) / 2,
        y_top * 0.95,
        f"trigger [{trig_start}, {trig_end})",
        ha="center",
        va="top",
        fontsize=9,
        color="#8B0000",
        fontweight="bold",
    )
    return handles, labels


def plot_scenario_timelines(rows: list[MethodRow], out_dir: Path, scene_name: str) -> None:
    if not rows:
        return

    mpl, plt = _get_matplotlib()
    with mpl.rc_context(PAPER_RC):
        plt.style.use("classic")
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(13.2, 6.4),
            sharex="col",
            gridspec_kw={"height_ratios": [1.0, 1.0], "wspace": 0.18, "hspace": 0.18},
        )
        handles_a, labels_a = _plot_one_timeline(
            axes[0, 0],
            axes[1, 0],
            rows,
            "scenario_A",
            "Scenario A: trigger at episode start",
        )
        handles_b, labels_b = _plot_one_timeline(
            axes[0, 1],
            axes[1, 1],
            rows,
            "scenario_B",
            "Scenario B: trigger at midpoint",
        )

        handles = handles_a or handles_b
        labels = labels_a or labels_b
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.04),
                ncol=min(5, len(handles)),
                frameon=False,
                handlelength=2.4,
                columnspacing=1.1,
                prop={"style": "italic", "weight": "bold", "size": 11},
            )
        fig.suptitle(scene_name, y=1.10, fontsize=14, fontweight="bold", fontstyle="italic")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        _save(fig, out_dir / "02_scenario_AB_timeline")


def build_table(rows: list[MethodRow], k_values: list[int]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in rows:
        entry = {
            "method": row.label,
            "key": row.key,
            "CR": row.cr,
            "CR_std": row.cr_std,
            "CR_t": row.cr_t,
            "CR_t_std": row.cr_t_std,
            "scenario_A_total_return": row.scenario_a_total,
            "scenario_B_total_return": row.scenario_b_total,
            "ASR_pct": row.asr_pct,
            "FTR_pct": row.ftr_pct,
            "scenario_A_win_ASR_pct": row.scenario_a_win_asr_pct,
            "scenario_A_post_ASR_pct": row.scenario_a_post_asr_pct,
            "scenario_B_win_ASR_pct": row.scenario_b_win_asr_pct,
            "scenario_B_post_ASR_pct": row.scenario_b_post_asr_pct,
        }
        for k in k_values:
            entry[f"K{k}_post_ASR_pct"] = row.post_asr.get(k)
        table.append(entry)
    return table


def write_summary_csv(table: list[dict[str, Any]], path: Path) -> None:
    if not table:
        return
    fields: list[str] = []
    for row in table:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    print(f"  saved {path}")


def print_table(table: list[dict[str, Any]]) -> None:
    print("method                 CR      CR_t    A-total  FTR%    ASR%")
    print("-" * 66)
    for row in table:
        a_total = row.get("scenario_A_total_return")
        print(
            f"{row['method'][:20]:<20} "
            f"{row.get('CR', 0):7.0f} "
            f"{row.get('CR_t', 0):7.0f} "
            f"{a_total if a_total is not None else float('nan'):7.0f} "
            f"{row.get('FTR_pct', 0):6.2f} "
            f"{row.get('ASR_pct', 0):6.1f}"
        )


def generate_comparison_figures(repo_root: Path, cfg: SceneConfig, out_dir: Path) -> int:
    rows, missing = load_rows(repo_root, cfg)
    if missing:
        print(f"[info] Skipping methods without eval_results.json: {', '.join(missing)}")
    if not rows:
        print("ERROR: No method results loaded.", file=sys.stderr)
        return 1

    table = build_table(rows, cfg.k_values)
    print(f"\n=== {cfg.name} plotted values ===")
    print_table(table)
    print()

    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(table, out_dir / "summary_table.csv")

    print(f"Writing figures -> {out_dir}")
    plot_return_grouped_bars(rows, out_dir, cfg.name)
    plot_scenario_timelines(rows, out_dir, cfg.name)

    readme = out_dir / "README_return_timeline.txt"
    readme.write_text(
        f"Return/timeline figures - {cfg.name}\n"
        "====================================\n"
        "01_return_grouped_bars.png/pdf:\n"
        "  grouped bars per method: clean return, full-trigger return, and\n"
        "  Scenario-A total return after a start-window trigger.\n\n"
        "02_scenario_AB_timeline.png/pdf:\n"
        "  Scenario A/B per-step reward and action-target cosine traces.\n\n"
        "summary_table.csv:\n"
        "  scalar values used for plotting plus FTR/ASR table columns.\n"
    )
    print(f"  saved {readme}")
    return 0


SCENE_PRESETS: dict[str, SceneConfig] = {
    "metaworld_reach": SceneConfig(
        name="MetaWorld Reach",
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
    "metaworld_drawer_open": SceneConfig(
        name="MetaWorld Drawer Open",
        method_paths={
            "causal": (
                "logdir/metaworld/backdoor/"
                "r2dreamer_drawer-open_ours_causal_open/eval/eval_results.json"
            ),
            "beat_adapted": (
                "logdir/metaworld/backdoor/"
                "r2dreamer_drawer-open_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_beat_adapted/"
                "eval/eval_results.json"
            ),
            "static_latent": (
                "logdir/metaworld/backdoor/"
                "r2dreamer_drawer-open_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_static_latent/"
                "eval/eval_results.json"
            ),
            "reward_only": (
                "logdir/metaworld/backdoor/"
                "r2dreamer_drawer-open_physical_pr0.3_a1.0_b0.0_lpi1.0_sk4_s0_reward_only/"
                "eval/eval_results.json"
            ),
            "reflective": (
                "logdir/metaworld/backdoor/"
                "r2dreamer_drawer-open_physical_pr0.3_a1.0_b1.0_lpi1.0_sk4_s0/eval/eval_results.json"
            ),
        },
        k_values=[1, 3, 5],
    ),
}
