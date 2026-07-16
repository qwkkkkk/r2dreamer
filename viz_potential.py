"""Latent target-action landscape visualization.

This script combines real-env trigger-withdrawal traces from eval_backdoor.py
and renders two comparable action-error landscapes:
  A. PCA back-projection through the frozen actor.
  B. Interpolation from real latent points.
"""

from __future__ import annotations

import json
import pathlib
import sys
import warnings
from dataclasses import dataclass

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd

from backdoor import BackdoorDreamer
from envs import make_envs

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
torch.set_float32_matmul_precision("high")


TRACE_ORDER = ("clean", "latent", "beat", "reflective", "ours")
MODEL_ORDER = TRACE_ORDER  # backward-compatible alias
MODEL_LABEL = {
    "clean": "Clean",
    "latent": "Latent-target",
    "beat": "BEAT-adapted",
    "reflective": "Reflective",
    "baseline": "Latent-target",
    "ours": "Ours (Causal)",
}
MODEL_COLOR = {
    "clean": "#303030",
    "latent": "#7A3DB8",
    "beat": "#D95F02",
    # Blue keeps Reflective visually far from red Ours when both attack well.
    "reflective": "#1F77B4",
    "baseline": "#6B3FA0",
    "ours": "#D62728",
}
TRACE_LINEWIDTH = {
    "clean": 1.15,
    "latent": 1.35,
    "beat": 1.35,
    "reflective": 1.35,
    "baseline": 2.0,
    "ours": 1.8,
}
TRACE_ZORDER = {
    "clean": 5,
    "latent": 7,
    "beat": 8,
    "reflective": 9,
    "baseline": 7,
    "ours": 10,
}


@dataclass
class PCA2:
    mean: np.ndarray
    components: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) @ self.components.T

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y @ self.components + self.mean


def _resolve(path_like, repo_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(str(path_like)).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_npz(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _scalar_str(x, default=""):
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    return default


def _fit_pca2(feats: np.ndarray) -> PCA2:
    feats = np.asarray(feats, dtype=np.float32)
    mean = feats.mean(axis=0, keepdims=True)
    centered = feats - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return PCA2(mean=mean.squeeze(0), components=vt[:2].astype(np.float32))


def _safe_path(value):
    return None if value is None or str(value).lower() in {"none", "null"} else value


def _resolve_trace_keys(viz):
    raw = getattr(viz, "trace_keys", None)
    if raw is None or str(raw).lower() in {"none", "null", ""}:
        return None
    if isinstance(raw, (list, tuple)):
        return [str(k) for k in raw]
    return [str(raw)]


def _resolve_trace_paths(viz) -> dict[str, str]:
    clean = _safe_path(viz.clean_trace)
    ours = _safe_path(viz.ours_trace)
    latent = _safe_path(getattr(viz, "latent_trace", None))
    beat = _safe_path(getattr(viz, "beat_trace", None))
    legacy = _safe_path(getattr(viz, "baseline_trace", None))
    if latent is None and legacy is not None:
        latent = legacy
    reflective = _safe_path(getattr(viz, "reflective_trace", None))
    paths = {"clean": clean, "latent": latent, "beat": beat, "ours": ours}
    if reflective is not None:
        paths["reflective"] = reflective
    missing = [k for k, p in paths.items() if p is None]
    if missing:
        raise SystemExit(f"Missing trace paths in config.viz: {missing}")
    return paths


def _ordered_trace_keys(traces) -> list[str]:
    """Stable legend order, skipping keys that were not provided."""
    return [k for k in TRACE_ORDER if k in traces]


def _trace_label(key: str) -> str:
    return MODEL_LABEL.get(key, key)


def _trace_color(key: str) -> str:
    return MODEL_COLOR.get(key, "#444444")


def _trace_lw(key: str) -> float:
    return float(TRACE_LINEWIDTH.get(key, 2.0))


def _trace_zorder(key: str) -> int:
    return int(TRACE_ZORDER.get(key, 6))


def _load_agent(config, actor_ckpt: pathlib.Path):
    print(f"[viz] loading actor/world-model shell from {actor_ckpt}")
    _, _, obs_space, act_space = make_envs(config.env)
    agent = BackdoorDreamer(config.model, obs_space, act_space, config.backdoor).to(config.device)
    act_dim = act_space.n if hasattr(act_space, "n") else int(sum(act_space.shape))
    tgt_cfg = config.backdoor.target_action
    target_action = [1.0] * act_dim if tgt_cfg is None else list(tgt_cfg)
    agent.set_target_action(target_action)
    ckpt = torch.load(actor_ckpt, map_location=config.device, weights_only=False)
    missing, unexpected = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")
    agent.clone_and_freeze()
    agent.eval()
    return agent


def _actor_phi(agent, feats: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    target = agent._target_action.detach().to(agent.device, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for start in range(0, len(feats), batch_size):
            chunk = torch.as_tensor(feats[start:start + batch_size], device=agent.device, dtype=torch.float32)
            action = agent._frozen_actor(chunk).mean
            out.append((action - target).norm(dim=-1).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _nan_gaussian_filter(field, sigma):
    if not sigma or float(sigma) <= 0:
        return field
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return field

    valid = np.isfinite(field).astype(np.float32)
    filled = np.nan_to_num(field, nan=0.0).astype(np.float32)
    num = gaussian_filter(filled, sigma=float(sigma))
    den = gaussian_filter(valid, sigma=float(sigma))
    out = np.full_like(field, np.nan, dtype=np.float32)
    mask = den > 1e-6
    out[mask] = num[mask] / den[mask]
    return out


def _knn_interpolate(points, values, grid_points, k=32, chunk=1024):
    points = np.asarray(points, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    grid_points = np.asarray(grid_points, dtype=np.float32)
    k = min(k, len(points))
    outs, densities = [], []
    for start in range(0, len(grid_points), chunk):
        gp = grid_points[start:start + chunk]
        d2 = ((gp[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1)
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dsel = np.take_along_axis(d2, idx, axis=1)
        vsel = values[idx]
        bandwidth = np.maximum(np.median(dsel, axis=1, keepdims=True), 1e-6)
        w = np.exp(-dsel / bandwidth) + 1e-8
        outs.append((w * vsel).sum(axis=1) / w.sum(axis=1))
        densities.append(np.sqrt(np.mean(dsel, axis=1)))
    return np.concatenate(outs, axis=0), np.concatenate(densities, axis=0)


def _interpolate_real_latent(pool_2d, phi, xx, yy, smooth_sigma=0.4,
                             knn_k=24, density_mask_quantile=0.85):
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    vals, density = _knn_interpolate(pool_2d, phi, grid_points, k=int(knn_k))
    field = vals.reshape(xx.shape).astype(np.float32)
    density = density.reshape(xx.shape).astype(np.float32)
    q = float(density_mask_quantile)
    if 0.0 < q < 1.0:
        cutoff = float(np.nanquantile(density, q))
        reliable = density <= cutoff
        field = np.where(reliable, field, np.nan).astype(np.float32)
    else:
        cutoff = float(np.nanmax(density))
        reliable = np.ones_like(field, dtype=bool)
    field = _nan_gaussian_filter(field, smooth_sigma).astype(np.float32)
    return field, density, reliable, cutoff


def _save_fig(fig, stem: pathlib.Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  saved {path}")


def _setup_axes(ax):
    ax.tick_params(direction="in", top=True, right=True, length=4.0, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")


def _trigger_bounds(trigger):
    idx = np.where(np.asarray(trigger, dtype=bool))[0]
    if not idx.size:
        return None, None
    return int(idx[0]), int(idx[-1]) + 1


def _event_keypoints(trigger, alive=None, T=None, post_stride=16, max_post=2):
    """Pick a small set of event-level trajectory points for uncluttered plots."""
    trigger = np.asarray(trigger, dtype=bool)
    T = int(T if T is not None else len(trigger))
    if T <= 0:
        return np.asarray([], dtype=int), None
    if alive is None:
        alive = np.ones(T, dtype=bool)
    else:
        alive = np.asarray(alive, dtype=bool)[:T]
        if len(alive) < T:
            alive = np.pad(alive, (0, T - len(alive)), constant_values=False)

    alive_idx = np.where(alive)[0]
    if not alive_idx.size:
        return np.asarray([0], dtype=int), None

    points = [int(alive_idx[0])]
    star_idx = None
    trig_idx = np.where(trigger[:T])[0]
    stride = max(1, int(post_stride))

    def add_nearest_alive(step):
        step = int(np.clip(step, 0, T - 1))
        if alive[step]:
            points.append(step)
            return
        later = alive_idx[alive_idx >= step]
        if later.size:
            points.append(int(later[0]))
            return
        earlier = alive_idx[alive_idx <= step]
        if earlier.size:
            points.append(int(earlier[-1]))

    if trig_idx.size:
        trig_on = int(trig_idx[0])
        trig_off = int(trig_idx[-1]) + 1
        mid = int((trig_on + trig_off - 1) // 2)
        add_nearest_alive(mid)
        star_idx = mid
        if trig_off < T:
            add_nearest_alive(trig_off)
        for k in range(1, max(1, int(max_post)) + 1):
            add_nearest_alive(trig_off + k * stride)
        add_nearest_alive(int(alive_idx[-1]))
    else:
        sparse = np.linspace(int(alive_idx[0]), int(alive_idx[-1]), 5).round().astype(int)
        for step in sparse[1:]:
            add_nearest_alive(int(step))

    points = np.asarray(sorted(set(p for p in points if 0 <= p < T and alive[p])), dtype=int)
    if points.size > 5 and trig_idx.size:
        keep = [points[0]]
        if star_idx is not None:
            star_pos = points[np.argmin(np.abs(points - int(star_idx)))]
            keep.append(int(star_pos))
        trig_off_pos = int(trig_idx[-1]) + 1
        if trig_off_pos < T:
            keep.append(int(points[np.argmin(np.abs(points - trig_off_pos))]))
        post = [int(p) for p in points if p not in keep and p > trig_off_pos]
        keep.extend(post[:1])
        keep.append(int(points[-1]))
        points = np.asarray(sorted(set(keep)), dtype=int)
    return points, star_idx


def _mark_trigger_window(ax, trigger):
    start, end = _trigger_bounds(trigger)
    if start is None:
        return
    ax.axvspan(start, end, color="#F2C230", alpha=0.13, linewidth=0)
    ax.axvline(start, color="#8B0000", linestyle="--", linewidth=1.1)
    ax.axvline(end, color="#8B0000", linestyle="--", linewidth=1.1)
    ymax = ax.get_ylim()[1]
    ax.text(start, ymax, "trigger on", ha="right", va="top", fontsize=8.5, color="#8B0000")
    ax.text(end, ymax, "off", ha="left", va="top", fontsize=8.5, color="#8B0000")


def _plot_traj(
    ax,
    xy,
    trigger,
    label,
    color,
    lw=2.0,
    zorder=5,
    alive=None,
    waypoint_stride=16,
    waypoint_labels=False,
    event_keypoints=True,
    show_full_trace=True,
    show_trigger_star=True,
):
    import matplotlib.patheffects as pe

    xy = np.asarray(xy, dtype=np.float32)
    trigger = np.asarray(trigger, dtype=bool)
    T = len(xy)
    if T == 0:
        return

    alive = np.ones(T, dtype=bool) if alive is None else np.asarray(alive, dtype=bool)[:T]
    if len(alive) < T:
        alive = np.pad(alive, (0, T - len(alive)), constant_values=False)

    halo = [pe.Stroke(linewidth=lw + 1.0, foreground="white", alpha=0.80), pe.Normal()]
    markevery = max(1, int(waypoint_stride))

    if event_keypoints:
        idx, star_idx = _event_keypoints(trigger, alive=alive, T=T, post_stride=markevery, max_post=2)
        if show_full_trace and T >= 2:
            faint_lw = max(0.55, lw * 0.58)
            for i in range(max(0, T - 1)):
                if not (alive[i] and alive[i + 1]):
                    continue
                line, = ax.plot(
                    xy[i:i + 2, 0],
                    xy[i:i + 2, 1],
                    color=color,
                    linewidth=faint_lw,
                    alpha=0.12,
                    solid_capstyle="round",
                    zorder=zorder - 1,
                )
                line.set_path_effects([
                    pe.Stroke(linewidth=faint_lw + 0.8, foreground="white", alpha=0.45),
                    pe.Normal(),
                ])
    else:
        idx = np.arange(0, T, markevery)
        if len(idx) == 0 or idx[-1] != T - 1:
            idx = np.unique(np.concatenate([idx, [T - 1]]))
        star_idx = None
        for i in range(max(0, T - 1)):
            if not (alive[i] and alive[i + 1]):
                continue
            alpha = 0.18 + 0.72 * (i + 1) / max(T - 1, 1)
            line, = ax.plot(
                xy[i:i + 2, 0],
                xy[i:i + 2, 1],
                color=color,
                linewidth=lw,
                alpha=alpha,
                solid_capstyle="round",
                zorder=zorder,
            )
            line.set_path_effects(halo)

    if len(idx) >= 2:
        for j, nxt in zip(idx[:-1], idx[1:]):
            if np.linalg.norm(xy[nxt] - xy[j]) <= 1e-8:
                continue
            ann = ax.annotate(
                "",
                xy=xy[nxt],
                xytext=xy[j],
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=max(0.70, lw * 0.66),
                    mutation_scale=6.5,
                    shrinkA=3.0,
                    shrinkB=3.0,
                ),
                zorder=zorder + 2,
            )
            ann.arrow_patch.set_path_effects([
                pe.Stroke(linewidth=max(1.25, lw * 0.66 + 0.85), foreground="white", alpha=0.78),
                pe.Normal(),
            ])

    for j in idx:
        ax.scatter(
            xy[j, 0],
            xy[j, 1],
            s=22 if label != "Clean" else 17,
            color=color,
            edgecolor="white",
            linewidth=0.65,
            alpha=0.96,
            zorder=zorder + 1,
        )
        if waypoint_labels and j % markevery == 0:
            ax.text(
                xy[j, 0],
                xy[j, 1],
                str(int(j)),
                fontsize=6.5,
                color="white",
                ha="center",
                va="center",
                zorder=zorder + 3,
            )

    if star_idx is not None and show_trigger_star and 0 <= int(star_idx) < T and alive[int(star_idx)]:
        j = int(star_idx)
        ax.scatter(
            xy[j, 0],
            xy[j, 1],
            marker="*",
            s=72,
            color="#F2C230",
            edgecolor="#111111",
            linewidth=0.65,
            zorder=13,
        )
    legend_line, = ax.plot([], [], color=color, linewidth=max(1.8, lw), label=label)
    legend_line.set_path_effects(halo)


def _display_smooth_field(field, sigma):
    """Display-only extra Gaussian smoothing; does not change saved landscape data."""
    if not sigma or float(sigma) <= 0:
        return field
    return _nan_gaussian_filter(np.asarray(field, dtype=np.float32), float(sigma)).astype(np.float32)


def _plot_potential(
    field,
    xx,
    yy,
    traces,
    out_stem,
    title,
    basin_threshold=None,
    reliable_mask=None,
    clip_quantile=0.98,
    levels=24,
    basin_mask_alpha=0.0,
    unreliable_mask_alpha=0.45,
    show_fine_contours=True,
    waypoint_stride=16,
    waypoint_labels=False,
    event_keypoints=True,
    show_full_trace=True,
    show_trigger_star=True,
    trace_keys=None,
    cbar_label=r"Actor target affinity ($-\Phi$); warmer = closer to $a^\dagger$",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9.5,
    })
    fig, ax = plt.subplots(figsize=(4.55, 4.15))
    finite = np.isfinite(field)
    cf = None
    cbar_lo = cbar_hi = None
    if finite.any():
        affinity = -np.asarray(field, dtype=np.float32)
        q = float(clip_quantile)
        if 0.5 < q < 1.0:
            cbar_lo, cbar_hi = np.nanquantile(affinity[finite], [1.0 - q, q])
            affinity = np.clip(affinity, cbar_lo, cbar_hi)
        affinity_plot = np.where(finite, affinity, np.nan)
        n_levels = int(levels)
        if cbar_lo is not None and cbar_hi is not None:
            level_edges = np.linspace(float(cbar_lo), float(cbar_hi), n_levels + 1)
        else:
            level_edges = np.linspace(
                float(np.nanmin(affinity_plot)),
                float(np.nanmax(affinity_plot)),
                n_levels + 1,
            )
        cmap = plt.get_cmap("RdYlBu_r", n_levels)
        norm = BoundaryNorm(level_edges, cmap.N, clip=True)
        cf = ax.contourf(
            xx,
            yy,
            affinity_plot,
            levels=level_edges,
            cmap=cmap,
            norm=norm,
            alpha=0.88,
            extend="neither",
            antialiased=True,
            zorder=0,
        )
        if show_fine_contours and n_levels >= 20:
            ax.contour(
                xx,
                yy,
                affinity_plot,
                levels=max(6, n_levels // 2),
                colors="#222222",
                linewidths=0.28,
                alpha=0.22,
                zorder=0.5,
            )
        if basin_threshold is not None:
            basin_in = np.where(finite, field <= float(basin_threshold), False).astype(np.float32)
            if float(basin_mask_alpha) > 0 and np.any(basin_in > 0.5):
                ax.contourf(
                    xx,
                    yy,
                    basin_in,
                    levels=[0.5, 1.5],
                    colors=["#F2DF8A"],
                    alpha=float(basin_mask_alpha),
                    zorder=1.5,
                )
            ax.contour(
                xx,
                yy,
                basin_in,
                levels=[0.5],
                colors="#7A0019",
                linewidths=1.1,
                linestyles="-",
                alpha=0.95,
                zorder=2.5,
            )
    else:
        cf = ax.contourf(xx, yy, np.zeros_like(field), levels=int(levels), cmap="RdYlBu_r", alpha=0.1)
    if reliable_mask is not None:
        unreliable = ~np.asarray(reliable_mask, dtype=bool)
        if unreliable.any():
            ax.contourf(
                xx,
                yy,
                np.where(unreliable, 1.0, np.nan),
                levels=[0.5, 1.5],
                colors=["#F7F7F7"],
                alpha=float(unreliable_mask_alpha),
                hatch="///",
                zorder=1.2,
            )
    plot_keys = list(trace_keys) if trace_keys else list(TRACE_ORDER)
    for key in plot_keys:
        if key not in traces:
            continue
        tr = traces[key]
        _plot_traj(
            ax,
            tr["xy"],
            tr["is_trigger"],
            _trace_label(key),
            _trace_color(key),
            lw=_trace_lw(key),
            zorder=_trace_zorder(key),
            alive=tr.get("alive_trace", None),
            waypoint_stride=waypoint_stride,
            waypoint_labels=waypoint_labels,
            event_keypoints=event_keypoints,
            show_full_trace=show_full_trace,
            show_trigger_star=show_trigger_star,
        )
    _setup_axes(ax)
    if title:
        ax.set_title(title)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.74,
        loc="best",
        handlelength=1.8,
        labelspacing=0.45,
        borderpad=0.35,
    )
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.035)
    if cbar_label:
        cbar.set_label(cbar_label)
    fig.tight_layout()
    _save_fig(fig, out_stem)
    plt.close(fig)


def _trace_arrays(tr, key="delta"):
    if key == "delta" and "delta_traces" in tr:
        arr = tr["delta_traces"].astype(np.float32)
        alive = tr.get("alive_traces", np.ones_like(arr, dtype=bool)).astype(bool)
    else:
        arr = tr["delta_trace"][None].astype(np.float32)
        alive = tr.get("alive_trace", np.ones_like(tr["delta_trace"], dtype=bool))[None].astype(bool)
    return arr, alive


def _mean_std(arr, alive):
    masked = np.where(alive, arr, np.nan)
    return np.nanmean(masked, axis=0), np.nanstd(masked, axis=0)


def _plot_delta_curve(traces, out_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for key in _ordered_trace_keys(traces):
        tr = traces[key]
        delta, alive = _trace_arrays(tr)
        mean, std = _mean_std(delta, alive)
        x = np.arange(len(mean))
        color = _trace_color(key)
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=_trace_lw(key),
            label=_trace_label(key),
            zorder=_trace_zorder(key),
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=_trace_zorder(key) - 1,
        )
    _mark_trigger_window(ax, traces["ours"]["is_trigger"])
    ax.set_xlabel("Real environment step after intervention")
    ax.set_ylabel(r"$\Delta(h)=||\pi_0(z_h)-a^\dagger||_2$")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_stem)
    plt.close(fig)


def _plot_clean_relative_persistence(traces, out_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean_delta, clean_alive = _trace_arrays(traces["clean"])
    clean_mean, _ = _mean_std(clean_delta, clean_alive)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for key in _ordered_trace_keys(traces):
        if key == "clean":
            continue
        delta, alive = _trace_arrays(traces[key])
        mean, std = _mean_std(delta, alive)
        L = min(len(clean_mean), len(mean))
        adv = clean_mean[:L] - mean[:L]
        color = _trace_color(key)
        ax.plot(
            np.arange(L),
            adv,
            color=color,
            linewidth=_trace_lw(key),
            label=_trace_label(key),
            zorder=_trace_zorder(key),
        )
        ax.fill_between(
            np.arange(L),
            adv - std[:L],
            adv + std[:L],
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=_trace_zorder(key) - 1,
        )
    _mark_trigger_window(ax, traces["ours"]["is_trigger"])
    ax.axhline(0, color="#222222", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Real environment step after intervention")
    ax.set_ylabel(r"Clean-relative affinity: $\Delta_{clean}(h)-\Delta(h)$")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_stem)
    plt.close(fig)


def _plot_basin_occupancy(traces, basin_threshold, out_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for key in _ordered_trace_keys(traces):
        delta, alive = _trace_arrays(traces[key])
        occ = np.where(alive, delta <= float(basin_threshold), np.nan).astype(np.float32)
        mean = np.nanmean(occ, axis=0)
        std = np.nanstd(occ, axis=0)
        color = _trace_color(key)
        x = np.arange(len(mean))
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=_trace_lw(key),
            label=_trace_label(key),
            zorder=_trace_zorder(key),
        )
        ax.fill_between(
            x,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=_trace_zorder(key) - 1,
        )
    first = traces["ours"]["is_trigger"]
    trig_idx = np.where(first)[0]
    if trig_idx.size:
        ax.axvline(int(trig_idx[-1]) + 1, color="#8B0000", linestyle="--", linewidth=1.2)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Real environment step after intervention")
    ax.set_ylabel("Target-basin occupancy")
    ax.set_title(rf"Basin: $\Delta(h)\leq {float(basin_threshold):.3f}$")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_stem)
    plt.close(fig)


def _post_mask(tr):
    trigger = np.asarray(tr["is_trigger"], dtype=bool)
    alive = np.asarray(tr.get("alive_trace", np.ones_like(trigger, dtype=bool)), dtype=bool)
    trig_idx = np.where(trigger)[0]
    if trig_idx.size:
        return alive & (np.arange(len(trigger)) >= int(trig_idx[-1]) + 1)
    return alive


def _compute_basin_threshold(traces, pool_phi, explicit=None, quantile=0.50, min_threshold=0.5):
    if explicit is not None and str(explicit).lower() not in {"none", "null"}:
        return float(explicit)
    vals = []
    for key in ("ours", "latent", "beat", "reflective"):
        if key not in traces:
            continue
        tr = traces[key]
        delta = np.asarray(tr["delta_trace"], dtype=np.float32)
        mask = np.asarray(tr["is_trigger"], dtype=bool)
        if mask.any():
            vals.append(delta[mask])
    if vals:
        base = np.concatenate(vals)
    else:
        base = np.asarray(pool_phi, dtype=np.float32)
    threshold = float(np.nanquantile(base, float(quantile)))
    if min_threshold is not None and str(min_threshold).lower() not in {"none", "null"}:
        threshold = max(threshold, float(min_threshold))
    return threshold


def _summarize(traces, phi_a, phi_b, out_dir, basin_threshold=None, phi_b_density=None, phi_b_reliable=None):
    summary = {
        "phi_A": {"min": float(np.nanmin(phi_a)), "max": float(np.nanmax(phi_a)), "mean": float(np.nanmean(phi_a))},
        "phi_B": {"min": float(np.nanmin(phi_b)), "max": float(np.nanmax(phi_b)), "mean": float(np.nanmean(phi_b))},
        "basin_threshold": None if basin_threshold is None else float(basin_threshold),
        "models": {},
    }
    if phi_b_density is not None:
        summary["phi_B_density"] = {
            "min": float(np.nanmin(phi_b_density)),
            "max": float(np.nanmax(phi_b_density)),
            "mean": float(np.nanmean(phi_b_density)),
        }
    if phi_b_reliable is not None:
        summary["phi_B_reliable_fraction"] = float(np.asarray(phi_b_reliable, dtype=bool).mean())

    clean_delta = np.asarray(traces["clean"]["delta_trace"], dtype=np.float32)
    for key, tr in traces.items():
        delta = np.asarray(tr["delta_trace"], dtype=np.float32)
        alive = np.asarray(tr.get("alive_trace", np.ones_like(delta, dtype=bool)), dtype=bool)
        post_mask = _post_mask(tr)
        post = delta[post_mask]
        x = np.arange(len(delta), dtype=np.float32)
        mask = post_mask & alive
        slope = 0.0
        if mask.sum() >= 2:
            xm = x[mask] - x[mask].mean()
            ym = delta[mask] - delta[mask].mean()
            slope = float((xm * ym).sum() / max(float((xm * xm).sum()), 1e-8))
        L = min(len(clean_delta), len(delta))
        rel = clean_delta[:L] - delta[:L]
        post_rel = rel[_post_mask(tr)[:L]]
        model_summary = {
            "delta0": float(delta[0]) if len(delta) else None,
            "post_delta_mean": float(post.mean()) if post.size else None,
            "post_delta_slope": slope,
            "post_clean_relative_mean": float(post_rel.mean()) if post_rel.size else None,
        }
        if basin_threshold is not None:
            model_summary["post_basin_occupancy"] = float((post <= float(basin_threshold)).mean()) if post.size else None
        summary["models"][key] = model_summary
    path = out_dir / "potential_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"  saved {path}")
    print(json.dumps(summary, indent=2))


@hydra.main(version_base=None, config_path="configs", config_name="configs_finetune")
def main(config):
    repo_root = pathlib.Path(get_original_cwd())
    viz = config.viz
    paths = _resolve_trace_paths(viz)
    trace_raw = {k: _load_npz(_resolve(p, repo_root)) for k, p in paths.items()}
    actor_value = _safe_path(getattr(viz, "actor_ckpt", None)) or config.ckpt_path
    actor_ckpt = pathlib.Path(str(actor_value)).expanduser()
    actor_ckpt = actor_ckpt if actor_ckpt.is_absolute() else repo_root / actor_ckpt
    agent = _load_agent(config, actor_ckpt)

    feat_blocks = []
    pool_blocks = []
    phi_blocks = []
    for raw in trace_raw.values():
        if "feat_traces" in raw:
            feat_blocks.append(raw["feat_traces"].reshape(-1, raw["feat_traces"].shape[-1]))
        else:
            feat_blocks.append(raw["feat_trace"])
        if "pool_feats" in raw and "pool_phi" in raw:
            pool_blocks.append(raw["pool_feats"])
            phi_blocks.append(raw["pool_phi"])
    all_feats = np.concatenate(feat_blocks + pool_blocks, axis=0).astype(np.float32)
    pca = _fit_pca2(all_feats)

    traces = {}
    for key, raw in trace_raw.items():
        feat = raw["feat_trace"].astype(np.float32)
        traces[key] = {
            "xy": pca.transform(feat),
            "feat_trace": feat,
            "delta_trace": raw["delta_trace"].astype(np.float32),
            "is_trigger": raw["is_trigger"].astype(bool),
            "alive_trace": raw.get("alive_trace", np.ones_like(raw["delta_trace"], dtype=bool)).astype(bool),
            "model_name": _scalar_str(raw.get("model_name", key), key),
        }
        if "delta_traces" in raw:
            traces[key]["delta_traces"] = raw["delta_traces"].astype(np.float32)
        if "alive_traces" in raw:
            traces[key]["alive_traces"] = raw["alive_traces"].astype(bool)

    pool_feats = np.concatenate(pool_blocks, axis=0).astype(np.float32) if pool_blocks else all_feats
    pool_phi = np.concatenate(phi_blocks, axis=0).astype(np.float32) if phi_blocks else _actor_phi(agent, pool_feats)
    pool_2d = pca.transform(pool_feats)

    all_2d = np.concatenate([pool_2d] + [traces[k]["xy"] for k in TRACE_ORDER], axis=0)
    pad = 0.08 * np.maximum(all_2d.max(axis=0) - all_2d.min(axis=0), 1e-6)
    lo = all_2d.min(axis=0) - pad
    hi = all_2d.max(axis=0) + pad
    grid_res = int(getattr(viz, "grid_res", 80))
    xs = np.linspace(lo[0], hi[0], grid_res)
    ys = np.linspace(lo[1], hi[1], grid_res)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    grid_feats = pca.inverse_transform(grid_points).astype(np.float32)
    phi_a = _actor_phi(agent, grid_feats).reshape(xx.shape)
    phi_b, phi_b_density, phi_b_reliable, phi_b_density_cutoff = _interpolate_real_latent(
        pool_2d,
        pool_phi,
        xx,
        yy,
        smooth_sigma=float(getattr(viz, "smooth_sigma", 0.4)),
        knn_k=int(getattr(viz, "knn_k", 24)),
        density_mask_quantile=float(getattr(viz, "density_mask_quantile", 0.85)),
    )
    basin_threshold = _compute_basin_threshold(
        traces,
        pool_phi,
        explicit=getattr(viz, "basin_threshold", None),
        quantile=float(getattr(viz, "basin_quantile", 0.50)),
        min_threshold=getattr(viz, "basin_min_threshold", 0.5),
    )

    out_dir_cfg = _safe_path(viz.output_dir)
    out_dir = _resolve(out_dir_cfg, repo_root) if out_dir_cfg is not None else repo_root / "figures" / "latent_potential"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "pca_landscape_data.npz",
        pca_mean=pca.mean,
        pca_components=pca.components,
        xx=xx,
        yy=yy,
        phi_A=phi_a,
        phi_B=phi_b,
        phi_B_density=phi_b_density,
        phi_B_reliable=phi_b_reliable,
        phi_B_density_cutoff=np.asarray(phi_b_density_cutoff, dtype=np.float32),
        basin_threshold=np.asarray(basin_threshold, dtype=np.float32),
        pool_2d=pool_2d,
        pool_phi=pool_phi,
    )
    trace_keys = _resolve_trace_keys(viz)
    show_trigger_star = bool(getattr(viz, "show_trigger_star", True))
    _plot_potential(phi_a, xx, yy, traces, out_dir / "potential_A_pca_backprojection",
                    "(A) PCA back-projection landscape",
                    basin_threshold=basin_threshold,
                    clip_quantile=float(getattr(viz, "potential_clip_quantile", 0.98)),
                    waypoint_stride=int(getattr(viz, "waypoint_stride", 16)),
                    waypoint_labels=bool(getattr(viz, "waypoint_labels", False)),
                    event_keypoints=bool(getattr(viz, "event_keypoints", True)),
                    show_full_trace=bool(getattr(viz, "show_full_trace", True)),
                    show_trigger_star=show_trigger_star,
                    trace_keys=trace_keys)
    _plot_potential(phi_b, xx, yy, traces, out_dir / "potential_B_real_latent_knn",
                    "(B) Real-latent KNN landscape",
                    basin_threshold=basin_threshold,
                    reliable_mask=phi_b_reliable,
                    clip_quantile=float(getattr(viz, "potential_clip_quantile", 0.92)),
                    levels=int(getattr(viz, "potential_b_levels", 14)),
                    basin_mask_alpha=float(getattr(viz, "basin_mask_alpha", 0.20)),
                    unreliable_mask_alpha=float(getattr(viz, "unreliable_mask_alpha", 0.30)),
                    show_fine_contours=False,
                    waypoint_stride=int(getattr(viz, "waypoint_stride", 16)),
                    waypoint_labels=bool(getattr(viz, "waypoint_labels", False)),
                    event_keypoints=bool(getattr(viz, "event_keypoints", True)),
                    show_full_trace=bool(getattr(viz, "show_full_trace", True)),
                    show_trigger_star=show_trigger_star,
                    trace_keys=trace_keys)
    _plot_delta_curve(traces, out_dir / "delta_curve")
    _plot_clean_relative_persistence(traces, out_dir / "clean_relative_persistence")
    _plot_basin_occupancy(traces, basin_threshold, out_dir / "basin_occupancy")
    _summarize(
        traces,
        phi_a,
        phi_b,
        out_dir,
        basin_threshold=basin_threshold,
        phi_b_density=phi_b_density,
        phi_b_reliable=phi_b_reliable,
    )


if __name__ == "__main__":
    main()
