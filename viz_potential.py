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


MODEL_ORDER = ("clean", "baseline", "ours")
MODEL_LABEL = {
    "clean": "Clean",
    "baseline": "Baseline",
    "ours": "Ours (Causal)",
}
MODEL_COLOR = {
    "clean": "#666666",
    "baseline": "#1679AB",
    "ours": "#B23A48",
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


def _idw_interpolate(points, values, grid_points, k=24, chunk=1024):
    points = np.asarray(points, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    grid_points = np.asarray(grid_points, dtype=np.float32)
    k = min(k, len(points))
    outs = []
    for start in range(0, len(grid_points), chunk):
        gp = grid_points[start:start + chunk]
        d2 = ((gp[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1)
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dsel = np.take_along_axis(d2, idx, axis=1)
        vsel = values[idx]
        w = 1.0 / (dsel + 1e-6)
        outs.append((w * vsel).sum(axis=1) / w.sum(axis=1))
    return np.concatenate(outs, axis=0)


def _interpolate_real_latent(pool_2d, phi, xx, yy, smooth_sigma=0.8):
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    try:
        from scipy.interpolate import griddata

        cubic = griddata(pool_2d, phi, (xx, yy), method="cubic")
        nearest = griddata(pool_2d, phi, (xx, yy), method="nearest")
        field = np.where(np.isnan(cubic), nearest, cubic)
    except Exception as exc:
        print(f"[viz] scipy griddata unavailable ({exc}); using IDW interpolation.")
        field = _idw_interpolate(pool_2d, phi, grid_points).reshape(xx.shape)

    if smooth_sigma and float(smooth_sigma) > 0:
        try:
            from scipy.ndimage import gaussian_filter

            field = gaussian_filter(field, sigma=float(smooth_sigma))
        except Exception:
            pass
    return field.astype(np.float32)


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


def _plot_traj(ax, xy, trigger, label, color, lw=2.0, zorder=5):
    xy = np.asarray(xy, dtype=np.float32)
    trigger = np.asarray(trigger, dtype=bool)
    T = len(xy)
    if T == 0:
        return
    for i in range(max(0, T - 1)):
        alpha = 0.25 + 0.7 * (i + 1) / max(T - 1, 1)
        ax.plot(xy[i:i + 2, 0], xy[i:i + 2, 1], color=color, linewidth=lw, alpha=alpha, zorder=zorder)
    markevery = max(1, T // 18)
    idx = np.arange(0, T, markevery)
    alphas = 0.25 + 0.7 * (idx + 1) / max(T, 1)
    for j, a in zip(idx, alphas):
        ax.scatter(xy[j, 0], xy[j, 1], s=26 if label != "Clean" else 18,
                   color=color, edgecolor="white", linewidth=0.45, alpha=float(a), zorder=zorder + 1)
    trig_idx = np.where(trigger)[0]
    if trig_idx.size:
        j = int(trig_idx[0])
        ax.scatter(xy[j, 0], xy[j, 1], marker="*", s=170, color="#F2C230",
                   edgecolor="#111111", linewidth=0.8, zorder=12)
    for frac in (0.28, 0.56, 0.82):
        j = min(max(0, int(frac * (T - 2))), max(T - 2, 0))
        if T >= 2 and np.linalg.norm(xy[j + 1] - xy[j]) > 1e-8:
            ax.annotate(
                "",
                xy=xy[j + 1],
                xytext=xy[j],
                arrowprops=dict(arrowstyle="->", color=color, lw=max(1.0, lw * 0.75), shrinkA=0, shrinkB=0),
                zorder=zorder + 2,
            )
    ax.plot([], [], color=color, linewidth=lw, label=label)


def _plot_potential(field, xx, yy, traces, out_stem, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
    })
    fig, ax = plt.subplots(figsize=(4.35, 4.05))
    levels = 24
    cf = ax.contourf(xx, yy, -field, levels=levels, cmap="RdYlBu_r", alpha=0.92)
    ax.contour(xx, yy, -field, levels=10, colors="#222222", linewidths=0.35, alpha=0.35)
    for key in MODEL_ORDER:
        tr = traces[key]
        _plot_traj(
            ax,
            tr["xy"],
            tr["is_trigger"],
            MODEL_LABEL[key],
            MODEL_COLOR[key],
            lw=3.0 if key == "ours" else (2.2 if key == "baseline" else 1.6),
            zorder=9 if key == "ours" else (7 if key == "baseline" else 5),
        )
    _setup_axes(ax)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", handlelength=2.2)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(r"Actor target affinity ($-\Phi$); warmer = closer to $a^\dagger$")
    fig.tight_layout()
    _save_fig(fig, out_stem)
    plt.close(fig)


def _plot_delta_curve(traces, out_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for key in MODEL_ORDER:
        tr = traces[key]
        if "delta_traces" in tr:
            delta = tr["delta_traces"].astype(np.float32)
            alive = tr.get("alive_traces", np.ones_like(delta, dtype=bool)).astype(bool)
            arr = np.where(alive, delta, np.nan)
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
        else:
            mean = tr["delta_trace"].astype(np.float32)
            std = np.zeros_like(mean)
        x = np.arange(len(mean))
        color = MODEL_COLOR[key]
        ax.plot(x, mean, color=color, linewidth=2.6 if key == "ours" else 1.9, label=MODEL_LABEL[key])
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    first = traces["ours"]["is_trigger"]
    trig_idx = np.where(first)[0]
    if trig_idx.size:
        ax.axvline(int(trig_idx[-1]) + 1, color="#8B0000", linestyle="--", linewidth=1.2)
        ax.text(int(trig_idx[-1]) + 1, ax.get_ylim()[1], "trigger off", ha="left", va="top", fontsize=9)
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


def _summarize(traces, phi_a, phi_b, out_dir):
    summary = {
        "phi_A": {"min": float(np.nanmin(phi_a)), "max": float(np.nanmax(phi_a)), "mean": float(np.nanmean(phi_a))},
        "phi_B": {"min": float(np.nanmin(phi_b)), "max": float(np.nanmax(phi_b)), "mean": float(np.nanmean(phi_b))},
        "models": {},
    }
    for key, tr in traces.items():
        delta = np.asarray(tr["delta_trace"], dtype=np.float32)
        trigger = np.asarray(tr["is_trigger"], dtype=bool)
        alive = np.asarray(tr.get("alive_trace", np.ones_like(delta, dtype=bool)), dtype=bool)
        post = delta[(~trigger) & alive]
        x = np.arange(len(delta), dtype=np.float32)
        mask = (~trigger) & alive
        slope = 0.0
        if mask.sum() >= 2:
            xm = x[mask] - x[mask].mean()
            ym = delta[mask] - delta[mask].mean()
            slope = float((xm * ym).sum() / max(float((xm * xm).sum()), 1e-8))
        summary["models"][key] = {
            "delta0": float(delta[0]) if len(delta) else None,
            "post_delta_mean": float(post.mean()) if post.size else None,
            "post_delta_slope": slope,
        }
    path = out_dir / "potential_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"  saved {path}")
    print(json.dumps(summary, indent=2))


@hydra.main(version_base=None, config_path="configs", config_name="configs_finetune")
def main(config):
    repo_root = pathlib.Path(get_original_cwd())
    viz = config.viz
    paths = {
        "clean": _safe_path(viz.clean_trace),
        "ours": _safe_path(viz.ours_trace),
        "baseline": _safe_path(viz.baseline_trace),
    }
    missing = [k for k, p in paths.items() if p is None]
    if missing:
        raise SystemExit(f"Missing trace paths in config.viz: {missing}")

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

    all_2d = np.concatenate([pool_2d] + [traces[k]["xy"] for k in MODEL_ORDER], axis=0)
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
    phi_b = _interpolate_real_latent(
        pool_2d,
        pool_phi,
        xx,
        yy,
        smooth_sigma=float(getattr(viz, "smooth_sigma", 0.8)),
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
        pool_2d=pool_2d,
        pool_phi=pool_phi,
    )
    _plot_potential(phi_a, xx, yy, traces, out_dir / "potential_A_pca_backprojection",
                    "(A) PCA back-projection landscape")
    _plot_potential(phi_b, xx, yy, traces, out_dir / "potential_B_real_latent_interpolation",
                    "(B) Real-latent interpolation landscape")
    _plot_delta_curve(traces, out_dir / "delta_curve")
    _summarize(traces, phi_a, phi_b, out_dir)


if __name__ == "__main__":
    main()
