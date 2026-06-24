"""Standalone offline evaluation for a stage-2 backdoored checkpoint.

Paper-level metrics reported:
    CR       Clean Return (mean ± std across envs)
    CR_t     Triggered Return (mean ± std) — full random-t* triggered rollout
    dR       CR - CR_t  (absolute return drop)
    dR_pct   dR / CR * 100%  (normalised drop)
    ASR      Attack Success Rate on triggered steps (mean ± std per env)
    FTR      False Trigger Rate on clean steps
    MSE      Action MSE on triggered steps

Fixed-window eval (two scenarios, with per-step breakdown):
    Scenario A: trigger from step 0   for eval_trig_K steps
    Scenario B: trigger from step eval_trig_start for eval_trig_K steps
    Each reports:
        pre_score      return before trigger window
        window_score   return during trigger window
        post_score     return after trigger window  (persistence impact)
        window_ASR     ASR during trigger window    (denom = window steps)
        post_ASR       ASR after trigger window     (RSSM persistence test)
        window_MSE     action MSE during trigger window
        per-step table of reward and cos_sim(action, a†)

Videos saved to <logdir>/eval/:
    eval_clean_video  — 10 clean episodes side by side
    eval_trig_video   — 10 triggered episodes side by side

Usage:
    python eval_backdoor.py \\
        --config-name configs_finetune \\
        env=dmc_vision env.task=dmc_ball_in_cup_catch \\
        ckpt_path=/path/to/backdoored/latest.pt \\
        env.eval_episode_num=10
"""

import json
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from backdoor import BackdoorDreamer, BackdoorTrainer
from envs import make_envs

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
torch.set_float32_matmul_precision("high")


class _EvalShim(BackdoorTrainer):
    """Reuses BackdoorTrainer rollout methods without building a replay buffer."""

    def __init__(self, eval_envs, backdoor_cfg):
        self.eval_envs = eval_envs
        self.trigger_type = str(getattr(backdoor_cfg, "trigger_type", "white"))
        self.trigger_size = int(backdoor_cfg.trigger_size)
        self.trigger_intensity = float(backdoor_cfg.trigger_intensity)
        self.trigger_eps = float(getattr(backdoor_cfg, "trigger_eps", 8)) / 255.0
        self.window_K = int(getattr(backdoor_cfg, "window_K", -1))
        self.eval_t_max = int(getattr(backdoor_cfg, "eval_t_max", 500))
        self.asr_threshold = float(getattr(backdoor_cfg, "asr_threshold", 0.9))
        self.asr_min_norm = float(getattr(backdoor_cfg, "asr_min_norm", 0.1))
        self.eval_trig_start = int(getattr(backdoor_cfg, "eval_trig_start", 250))
        self.eval_trig_K = int(getattr(backdoor_cfg, "eval_trig_K", 16))
        self.asr_vs_k = [int(k) for k in getattr(backdoor_cfg, "asr_vs_k", [1, 3, 5])]
        self.save_latent_traces = bool(getattr(backdoor_cfg, "save_latent_traces", True))


def _fixed_window_stats(out, trig_start, trig_K, n_envs, bar):
    """Print and collect stats for one fixed-window rollout."""
    trig_end = trig_start + trig_K
    w_steps = out["window_steps"].sum().clamp_min(1)
    p_steps = out["post_steps"].sum().clamp_min(1)

    pre_score    = out["pre_returns"].mean().item()
    win_score    = out["window_returns"].mean().item()
    post_score   = out["post_returns"].mean().item()
    win_score_std  = out["window_returns"].std().item()
    post_score_std = out["post_returns"].std().item()

    per_env_w_asr = out["window_hit"] / out["window_steps"].clamp_min(1)
    per_env_p_asr = out["post_hit"]   / out["post_steps"].clamp_min(1)
    w_asr     = per_env_w_asr.mean().item()
    w_asr_std = per_env_w_asr.std().item()
    p_asr     = per_env_p_asr.mean().item()
    p_asr_std = per_env_p_asr.std().item()
    w_mse     = (out["window_sq_err"].sum() / w_steps).item()

    dR_win  = pre_score - win_score
    dR_post = pre_score - post_score

    print(f"  Pre-window score       : {pre_score:8.2f}  (steps 0 – {trig_start-1})")
    print(f"  Window score           : {win_score:8.2f}  ± {win_score_std:.2f}"
          f"  (steps {trig_start} – {trig_end-1},  drop={dR_win:.1f})")
    print(f"  Post-window score      : {post_score:8.2f}  ± {post_score_std:.2f}"
          f"  (steps {trig_end} – end,  drop={dR_post:.1f})")
    print(f"  Window  ASR            : {w_asr*100:7.2f}%  ± {w_asr_std*100:.2f}%"
          f"  [denom=window steps,  K={trig_K}]")
    print(f"  Post-window ASR (persist): {p_asr*100:5.2f}%  ± {p_asr_std*100:.2f}%"
          f"  [RSSM persistence]")
    print(f"  Window  MSE            : {w_mse:8.4f}")
    print(bar)

    d = {
        "trig_start":   trig_start,
        "trig_K":       trig_K,
        "pre_score":    pre_score,
        "win_score":    win_score,    "win_score_std":  win_score_std,
        "post_score":   post_score,   "post_score_std": post_score_std,
        "dR_win":       dR_win,
        "dR_post":      dR_post,
        "win_ASR":      w_asr,        "win_ASR_std":    w_asr_std,
        "post_ASR":     p_asr,        "post_ASR_std":   p_asr_std,
        "win_MSE":      w_mse,
    }

    if "per_step_reward" in out:
        # Mean over envs (B dim), list of T floats
        ps_rew = out["per_step_reward"].mean(dim=1).tolist()
        ps_cos = out["per_step_cossim"].mean(dim=1).tolist()
        d["per_step_reward"] = ps_rew
        d["per_step_cossim"] = ps_cos

        # Print a compact per-zone summary table
        T = len(ps_rew)
        print(f"  Step-by-step summary (mean over {n_envs} envs):")
        print(f"  {'step':>6}  {'reward':>8}  {'cos_sim':>8}  zone")
        zones = ["pre", "window", "post"]
        prev_zone = None
        for t in range(T):
            z = ("window" if trig_start <= t < trig_end
                 else ("pre" if t < trig_start else "post"))
            if z != prev_zone:
                # Print one representative line per zone (first step)
                print(f"  {t:>6}  {ps_rew[t]:>8.3f}  {ps_cos[t]:>8.4f}  ← {z} starts")
                prev_zone = z
            elif t == T - 1:
                print(f"  {t:>6}  {ps_rew[t]:>8.3f}  {ps_cos[t]:>8.4f}")
        print(bar)

    return d


@hydra.main(version_base=None, config_path="configs", config_name="configs_finetune")
def main(config):
    tools.set_seed_everywhere(config.seed)

    logdir = pathlib.Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(logdir)

    print("Create envs (eval only).")
    _, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Build agent shell.")
    agent = BackdoorDreamer(
        config.model,
        obs_space,
        act_space,
        config.backdoor,
    ).to(config.device)

    act_dim = act_space.n if hasattr(act_space, "n") else int(sum(act_space.shape))
    tgt_cfg = config.backdoor.target_action
    target_action = [1.0] * act_dim if tgt_cfg is None else list(tgt_cfg)
    agent.set_target_action(target_action)

    print(f"Load checkpoint: {config.ckpt_path}")
    ckpt = torch.load(pathlib.Path(config.ckpt_path).expanduser(),
                      map_location=config.device, weights_only=False)
    missing, unexpected = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")
    agent.clone_and_freeze()
    agent.eval()

    shim = _EvalShim(eval_envs, config.backdoor)
    n_envs = eval_envs.env_num
    trig_K = shim.eval_trig_K
    trig_mid = shim.eval_trig_start

    bar = "=" * 64

    trigger_type = shim.trigger_type

    # ── 1. Full random-t* triggered rollout (matches training distribution) ─────
    if trigger_type == "physical":
        print(f"\nRolling out {n_envs} clean episodes (physical trigger: OFF throughout) ...")
    else:
        print(f"\nRolling out {n_envs} clean episodes ...")
    clean = shim._run_eval_rollout(agent, apply_trigger=False, collect_video=True)

    if trigger_type == "physical":
        print(f"Rolling out {n_envs} full-trigger episodes "
              f"(physical trigger: ON for full episode) ...")
    else:
        print(f"Rolling out {n_envs} full-trigger episodes "
              f"(random t*, window_K={shim.window_K}) ...")
    trig  = shim._run_eval_rollout(agent, apply_trigger=True,  collect_video=True)

    clean_steps = clean["step_count"].sum().clamp_min(1)
    trig_steps  = trig["step_count"].sum().clamp_min(1)
    per_env_asr = trig["hit_count"] / trig["step_count"].clamp_min(1)

    cr        = clean["returns"].mean().item()
    cr_std    = clean["returns"].std().item()
    cr_trig   = trig["returns"].mean().item()
    cr_t_std  = trig["returns"].std().item()
    asr       = per_env_asr.mean().item()
    asr_std   = per_env_asr.std().item()
    ftr       = (clean["hit_count"].sum() / clean_steps).item()
    act_mse   = (trig["sq_err_sum"].sum() / trig_steps).item()
    dR        = cr - cr_trig
    dR_pct    = dR / max(abs(cr), 1e-8) * 100.0

    print()
    print(bar)
    print(f"  Task: {config.env.task}  |  envs: {n_envs}  |  target_action = {target_action}")
    print(f"  ckpt: {config.ckpt_path}")
    print(bar)
    print(f"  [Full random-t* triggered rollout]")
    print(f"  Clean Return   (CR)     : {cr:8.2f}  ± {cr_std:.2f}")
    print(f"  Trigger Return (CR_t)   : {cr_trig:8.2f}  ± {cr_t_std:.2f}")
    print(f"  Return Drop    (dR)     : {dR:8.2f}  ({dR_pct:.1f}% of CR)")
    print(f"  Attack Success (ASR)    : {asr*100:7.2f}%  ± {asr_std*100:.2f}%"
          f"  (cos>{shim.asr_threshold}, ||a||>={shim.asr_min_norm})")
    print(f"  False Trigger  (FTR)    : {ftr*100:7.2f}%")
    print(f"  Action MSE     (MSE)    : {act_mse:8.4f}")
    print(bar)

    _phys_win_label = "physical_window" if trigger_type == "physical" else "pixel_window"

    results = {
        "ckpt": str(config.ckpt_path),
        "task": config.env.task,
        "n_envs": n_envs,
        "CR": cr,       "CR_std": cr_std,
        "CR_t": cr_trig, "CR_t_std": cr_t_std,
        "dR": dR,        "dR_pct": dR_pct,
        "ASR": asr,      "ASR_std": asr_std,
        "FTR": ftr,
        "MSE": act_mse,
        "trigger_eval": {
            "trigger_type": trigger_type,
            "full_rollout_mode": (
                "physical_full_episode" if trigger_type == "physical"
                else "windowed_pixel"
            ),
            "scenario_A": {
                "mode": _phys_win_label,
                "trig_start": 0,
                "trig_K": trig_K,
            },
            "scenario_B": {
                "mode": _phys_win_label,
                "trig_start": trig_mid,
                "trig_K": trig_K,
            },
        },
    }

    # ── 2. Fixed-window eval, Scenario A: trigger from step 0 ────────────────
    if trigger_type == "physical":
        print(f"\nRolling out {n_envs} episodes — Scenario A: "
              f"physical trigger active only on steps [0, {trig_K}) ...")
    else:
        print(f"\nRolling out {n_envs} episodes — Scenario A: trigger steps 0 – {trig_K-1} ...")
    out_a = shim._run_fixed_trigger_rollout(agent, trig_start=0, trig_K=trig_K,
                                            collect_perstep=True,
                                            collect_video=True)
    print()
    print(bar)
    print(f"  [Fixed window A: trigger @ steps 0 – {trig_K-1}, K={trig_K}]")
    sa = _fixed_window_stats(out_a, trig_start=0, trig_K=trig_K, n_envs=n_envs, bar=bar)
    sa["mode"] = _phys_win_label
    results["scenario_A"] = sa

    # ── 3. Fixed-window eval, Scenario B: trigger from midpoint ──────────────
    if trigger_type == "physical":
        print(f"\nRolling out {n_envs} episodes — Scenario B: "
              f"physical trigger active only on steps [{trig_mid}, {trig_mid+trig_K}) ...")
    else:
        print(f"\nRolling out {n_envs} episodes — Scenario B: "
              f"trigger steps {trig_mid} – {trig_mid+trig_K-1} ...")
    out_b = shim._run_fixed_trigger_rollout(agent, trig_start=trig_mid, trig_K=trig_K,
                                            collect_perstep=True,
                                            collect_video=True)
    print()
    print(bar)
    print(f"  [Fixed window B: trigger @ steps {trig_mid} – {trig_mid+trig_K-1}, K={trig_K}]")
    sb = _fixed_window_stats(out_b, trig_start=trig_mid, trig_K=trig_K, n_envs=n_envs, bar=bar)
    sb["mode"] = _phys_win_label
    results["scenario_B"] = sb

    # --- ASR-vs-K persistence probe: trigger from step 0, then withdraw ---
    asr_vs_k = {}
    latent_traces = {}
    for k_probe in shim.asr_vs_k:
        print(f"\nRolling out {n_envs} episodes - ASR-vs-K probe: trigger steps [0, {k_probe}) ...")
        out_k = shim._run_fixed_trigger_rollout(
            agent,
            trig_start=0,
            trig_K=int(k_probe),
            collect_perstep=True,
            collect_latent_trace=shim.save_latent_traces,
        )
        print()
        print(bar)
        print(f"  [ASR-vs-K: trigger @ steps 0-{int(k_probe)-1}, K={int(k_probe)}]")
        sk = _fixed_window_stats(out_k, trig_start=0, trig_K=int(k_probe), n_envs=n_envs, bar=bar)
        sk["mode"] = _phys_win_label
        asr_vs_k[str(int(k_probe))] = sk
        if shim.save_latent_traces and "latent_feat" in out_k:
            latent_traces[str(int(k_probe))] = out_k["latent_feat"]
    results["asr_vs_k"] = asr_vs_k

    if latent_traces:
        latent_path = logdir / "latent_traces.pt"
        torch.save(latent_traces, latent_path)
        print(f"Latent traces saved to {latent_path}")

    # ── 4. Clean per-step rollout for plot baseline (trigger never fires) ─────
    # trig_start is set far beyond any episode length so in_window is always False.
    # For physical trigger this is a no-op (trigger stays off, no pixel modification).
    print(f"\nRolling out {n_envs} episodes — Clean per-step baseline "
          f"({'physical trigger stays OFF' if trigger_type == 'physical' else 'no pixel trigger'}) ...")
    out_clean_ps = shim._run_fixed_trigger_rollout(
        agent, trig_start=99999, trig_K=1, collect_perstep=True)

    # ── Save results JSON ─────────────────────────────────────────────────────
    out_json = logdir / "eval_results.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # ── Save videos to TensorBoard ────────────────────────────────────────────
    if clean["video"] is not None:
        logger.video("eval_clean_video", tools.to_np(clean["video"]))
    if trig["video"] is not None:
        logger.video("eval_trig_video", tools.to_np(trig["video"]))
    if out_a.get("video") is not None:
        logger.video("eval_scenario_A_video", tools.to_np(out_a["video"]))
    if out_b.get("video") is not None:
        logger.video("eval_scenario_B_video", tools.to_np(out_b["video"]))
    logger.write(0)
    print(f"Videos saved to {logdir} (open with: tensorboard --logdir {logdir})")

    # ── Save eval artifacts (plots + individual mp4s + CSV + trigger visuals) ─
    _save_eval_artifacts(logdir, clean, trig, out_clean_ps, results, n_envs,
                         scenario_a_rollout=out_a, scenario_b_rollout=out_b)
    _save_trigger_visuals(logdir, agent, config.backdoor, clean, trig)


# ══════════════════════════════════════════════════════════════════════════════
# Artifact export helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save_videos_mp4(video_np, out_dir, prefix, fps=16):
    """Save each env's trajectory as an individual mp4.

    Args:
        video_np: (B, T, H, W, C) uint8 numpy array
        out_dir:  pathlib.Path, directory to write into
        prefix:   filename prefix, e.g. 'clean' or 'triggered'
        fps:      playback fps (16 = 1 agent-step per frame at action_repeat=2)
    """
    try:
        import imageio
    except ImportError:
        print("  [warn] imageio not installed — skipping mp4 export (pip install imageio[ffmpeg])")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    B = video_np.shape[0]
    for b in range(B):
        frames = video_np[b]  # (T, H, W, C)
        path = str(out_dir / f"{prefix}_env{b:02d}.mp4")
        writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                    output_params=["-crf", "18"])
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    print(f"  Saved {B} mp4s  →  {out_dir}/{prefix}_env*.mp4")


def _plot_reward_cossim(out, label, color, trig_start, trig_K, clean_rew, ax_rew, ax_cos):
    """Draw reward + cos_sim curves for one fixed-window scenario onto given axes."""
    import numpy as np

    trig_end = trig_start + trig_K
    ps_rew = np.array(out["per_step_reward"])  # (T,) already mean-over-envs from JSON
    ps_cos = np.array(out["per_step_cossim"])
    T = len(ps_rew)
    steps = np.arange(T)

    ax_rew.plot(steps, ps_rew, color=color, linewidth=1.2, label=label)
    if clean_rew is not None:
        ax_rew.plot(steps, np.array(clean_rew), color="steelblue",
                    linewidth=1.0, alpha=0.6, label="clean")
    ax_rew.axvspan(trig_start, trig_end, alpha=0.12, color="red",
                   label=f"trigger [{trig_start}, {trig_end})")
    ax_rew.set_ylabel("Reward")
    ax_rew.legend(fontsize=8)
    ax_rew.grid(alpha=0.3)

    ax_cos.plot(steps, ps_cos, color=color, linewidth=1.2)
    ax_cos.axvspan(trig_start, trig_end, alpha=0.12, color="red")
    ax_cos.axhline(0.9,  color="gray", linestyle="--", linewidth=0.8,
                   label="ASR threshold (0.9)")
    ax_cos.axhline(0.0,  color="black", linestyle="-",  linewidth=0.4, alpha=0.4)
    ax_cos.set_ylabel("cos_sim(a, a†)")
    ax_cos.set_xlabel("Step")
    ax_cos.legend(fontsize=8)
    ax_cos.grid(alpha=0.3)


def _save_trigger_visuals(logdir, agent, backdoor_cfg, clean_rollout, trig_rollout=None):
    """Save example original / trigger / triggered-observation images.

    For invis: trigger image = signed delta around mid-gray.
    For white: trigger image = black canvas with white patch.
    For physical: side-by-side of env-rendered clean vs. triggered frame (no delta to show).
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    video = clean_rollout.get("video")
    if video is None:
        print("  [warn] no clean video available - skipping trigger visual export")
        return

    video_np = tools.to_np(video)
    if video_np.ndim != 5 or video_np.shape[0] == 0 or video_np.shape[1] == 0:
        print("  [warn] unexpected clean video shape - skipping trigger visual export")
        return

    vis_dir = logdir / "trigger_visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)

    obs = video_np[0, 0]
    if obs.dtype != np.uint8:
        obs = np.clip(obs, 0, 255).astype(np.uint8)

    trigger_type = str(getattr(backdoor_cfg, "trigger_type", "white"))
    H, W, C = obs.shape

    if trigger_type == "physical":
        # For physical trigger the modification is in the rendered frame itself.
        # Use the first frame of the triggered rollout video as the triggered obs.
        trig_video = None if trig_rollout is None else trig_rollout.get("video")
        if trig_video is not None:
            trig_np = tools.to_np(trig_video)
            triggered = trig_np[0, 0]
            if triggered.dtype != np.uint8:
                triggered = np.clip(triggered, 0, 255).astype(np.uint8)
        else:
            triggered = obs.copy()  # fallback: no triggered video available

        # Pixel diff as the "trigger visualization".
        diff = np.abs(triggered.astype(np.int32) - obs.astype(np.int32)).astype(np.uint8)
        trigger_vis = np.clip(diff * 4, 0, 255).astype(np.uint8)  # amplify for visibility
        trigger_title = "pixel diff (clean vs. triggered) ×4"

        plt.imsave(vis_dir / "original_obs.png", obs)
        plt.imsave(vis_dir / "trigger_visualization.png", trigger_vis)
        plt.imsave(vis_dir / "triggered_obs.png", triggered)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, img, title in [
            (axes[0], obs, "clean observation"),
            (axes[1], trigger_vis, trigger_title),
            (axes[2], triggered, "triggered observation"),
        ]:
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(vis_dir / "trigger_triplet.png", dpi=200)
        plt.close(fig)
        print(f"  Trigger visuals saved: {vis_dir}")
        return

    if trigger_type == "invis" and getattr(agent, "delta", None) is not None:
        eps = float(getattr(agent, "trigger_eps", 8.0 / 255.0))
        delta = tools.to_np(agent.delta.detach().cpu().clamp(-eps, eps))
        if delta.shape != obs.shape:
            print(f"  [warn] delta shape {delta.shape} != obs shape {obs.shape} - skipping trigger visuals")
            return
        obs_f = obs.astype(np.float32) / 255.0
        trig_f = np.clip(obs_f + delta, 0.0, 1.0)
        triggered = (trig_f * 255.0).round().astype(np.uint8)
        trigger_vis = np.clip(delta / max(eps, 1e-8) * 0.5 + 0.5, 0.0, 1.0)
        trigger_vis = (trigger_vis * 255.0).round().astype(np.uint8)
        trigger_title = f"trigger delta (scaled, eps={eps:.4f})"
    else:
        size = int(getattr(backdoor_cfg, "trigger_size", 8))
        intensity = float(getattr(backdoor_cfg, "trigger_intensity", 1.0))
        triggered = obs.copy()
        val = int(round(np.clip(intensity, 0.0, 1.0) * 255.0))
        triggered[-size:, -size:, :] = val
        trigger_vis = np.zeros_like(obs)
        trigger_vis[-size:, -size:, :] = val
        trigger_title = f"white patch ({size}x{size})"

    plt.imsave(vis_dir / "original_obs.png", obs)
    plt.imsave(vis_dir / "trigger_visualization.png", trigger_vis)
    plt.imsave(vis_dir / "triggered_obs.png", triggered)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, title in [
        (axes[0], obs, "original observation"),
        (axes[1], trigger_vis, trigger_title),
        (axes[2], triggered, "observation + trigger"),
    ]:
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(vis_dir / "trigger_triplet.png", dpi=200)
    plt.close(fig)
    print(f"  Trigger visuals saved: {vis_dir}")


def _save_eval_artifacts(logdir, clean_rollout, trig_rollout,
                         out_clean_ps, results, n_envs,
                         scenario_a_rollout=None, scenario_b_rollout=None):
    """Write all visual and tabular artifacts to <logdir>/eval/.

    Structure created:
        <logdir>/eval/
            videos/
                clean_env00.mp4 … clean_env09.mp4
                triggered_env00.mp4 … triggered_env09.mp4
            plots/
                scenario_A.png        reward + cos_sim, trigger from step 0
                scenario_B.png        reward + cos_sim, trigger from midpoint
                metrics_bar.png       bar chart of headline metrics
            metrics_summary.csv       all scalar results
    """
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_dir = logdir
    eval_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = eval_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    vid_dir  = eval_dir / "videos"

    print(f"\nSaving eval artifacts to {eval_dir} ...")

    # ── 1. Individual mp4 videos ──────────────────────────────────────────────
    if clean_rollout.get("video") is not None:
        _save_videos_mp4(tools.to_np(clean_rollout["video"]),
                         vid_dir, prefix="clean")
    if trig_rollout.get("video") is not None:
        _save_videos_mp4(tools.to_np(trig_rollout["video"]),
                         vid_dir, prefix="triggered")
    if scenario_a_rollout is not None and scenario_a_rollout.get("video") is not None:
        _save_videos_mp4(tools.to_np(scenario_a_rollout["video"]),
                         vid_dir, prefix="scenario_A")
    if scenario_b_rollout is not None and scenario_b_rollout.get("video") is not None:
        _save_videos_mp4(tools.to_np(scenario_b_rollout["video"]),
                         vid_dir, prefix="scenario_B")

    # ── 2. Reward + cos_sim curves ────────────────────────────────────────────
    # Clean per-step trace: mean over envs from the no-trigger fixed-window rollout.
    # per_step_reward shape is (T, B); take mean over B.
    clean_rew_trace = None
    if "per_step_reward" in out_clean_ps:
        ps = out_clean_ps["per_step_reward"]  # tensor (T, B)
        clean_rew_trace = ps.float().mean(dim=1).tolist()

    # results["scenario_A/B"] already contain trig_start, trig_K, per_step_reward,
    # per_step_cossim as plain lists (mean over envs, computed by _fixed_window_stats).
    sc_b_start = results.get("scenario_B", {}).get("trig_start", 250)
    for scenario_key, out, label, color, fname in [
        ("scenario_A", results.get("scenario_A", {}),
         "triggered (from step 0)", "#d62728", "scenario_A.png"),
        ("scenario_B", results.get("scenario_B", {}),
         f"triggered (from step {sc_b_start})", "#ff7f0e", "scenario_B.png"),
    ]:
        if "per_step_reward" not in out:
            continue
        fig, (ax_rew, ax_cos) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        fig.suptitle(
            f"{results.get('task', '')}  —  {scenario_key}  "
            f"(K={out['trig_K']}, trigger [{out['trig_start']}, "
            f"{out['trig_start'] + out['trig_K']})",
            fontsize=11,
        )
        _plot_reward_cossim(
            out, label, color,
            trig_start=out["trig_start"], trig_K=out["trig_K"],
            clean_rew=clean_rew_trace,
            ax_rew=ax_rew, ax_cos=ax_cos,
        )
        plt.tight_layout()
        plt.savefig(plot_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {plot_dir / fname}")

    # ── 3. Metrics bar chart ──────────────────────────────────────────────────
    bar_specs = [
        ("CR",      results.get("CR",    0), results.get("CR_std",    0), "#4c72b0", "Clean Return"),
        ("CR_t",    results.get("CR_t",  0), results.get("CR_t_std",  0), "#dd8452", "Triggered Return"),
        ("ASR %",   results.get("ASR",   0) * 100, results.get("ASR_std", 0) * 100, "#c44e52", "ASR (%)"),
        ("FTR %",   results.get("FTR",   0) * 100, 0,                               "#937860", "FTR (%)"),
        ("dR",      results.get("dR",    0), 0,                                      "#8172b2", "Return Drop"),
    ]
    labels = [s[0] for s in bar_specs]
    vals   = [s[1] for s in bar_specs]
    errs   = [s[2] for s in bar_specs]
    colors = [s[3] for s in bar_specs]
    descs  = [s[4] for s in bar_specs]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, yerr=errs, capsize=5, color=colors, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(descs, fontsize=10)
    ax.set_title(f"Eval Metrics — {results.get('task', '')}  (n_envs={n_envs})",
                 fontsize=11)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(abs(v) for v in vals) * 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "metrics_bar.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {plot_dir / 'metrics_bar.png'}")

    # ── 4. Metrics CSV ────────────────────────────────────────────────────────
    csv_path = eval_dir / "metrics_summary.csv"
    scalar_rows = [
        ("task",      results.get("task", "")),
        ("ckpt",      results.get("ckpt", "")),
        ("n_envs",    results.get("n_envs", "")),
        ("CR",        results.get("CR",       "")),
        ("CR_std",    results.get("CR_std",   "")),
        ("CR_t",      results.get("CR_t",     "")),
        ("CR_t_std",  results.get("CR_t_std", "")),
        ("dR",        results.get("dR",       "")),
        ("dR_pct",    results.get("dR_pct",   "")),
        ("ASR",       results.get("ASR",      "")),
        ("ASR_std",   results.get("ASR_std",  "")),
        ("FTR",       results.get("FTR",      "")),
        ("MSE",       results.get("MSE",      "")),
        # scenario A
        ("A_win_ASR",   results.get("scenario_A", {}).get("win_ASR",  "")),
        ("A_post_ASR",  results.get("scenario_A", {}).get("post_ASR", "")),
        ("A_win_score", results.get("scenario_A", {}).get("win_score","")),
        ("A_post_score",results.get("scenario_A", {}).get("post_score","")),
        ("A_win_MSE",   results.get("scenario_A", {}).get("win_MSE",  "")),
        # scenario B
        ("B_pre_score", results.get("scenario_B", {}).get("pre_score", "")),
        ("B_win_ASR",   results.get("scenario_B", {}).get("win_ASR",   "")),
        ("B_post_ASR",  results.get("scenario_B", {}).get("post_ASR",  "")),
        ("B_win_score", results.get("scenario_B", {}).get("win_score", "")),
        ("B_post_score",results.get("scenario_B", {}).get("post_score","")),
        ("B_dR_win",    results.get("scenario_B", {}).get("dR_win",    "")),
        ("B_win_MSE",   results.get("scenario_B", {}).get("win_MSE",   "")),
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for row in scalar_rows:
            writer.writerow(row)
    print(f"  CSV  saved: {csv_path}")
    print(f"Artifacts complete.")


if __name__ == "__main__":
    main()
