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

    # ── 1. Full random-t* triggered rollout (matches training distribution) ─────
    print(f"\nRolling out {n_envs} clean episodes ...")
    clean = shim._run_eval_rollout(agent, apply_trigger=False, collect_video=True)
    print(f"Rolling out {n_envs} full-trigger episodes (random t*, window_K={shim.window_K}) ...")
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
    }

    # ── 2. Fixed-window eval, Scenario A: trigger from step 0 ────────────────
    print(f"\nRolling out {n_envs} episodes — Scenario A: trigger steps 0 – {trig_K-1} ...")
    out_a = shim._run_fixed_trigger_rollout(agent, trig_start=0, trig_K=trig_K,
                                            collect_perstep=True)
    print()
    print(bar)
    print(f"  [Fixed window A: trigger @ steps 0 – {trig_K-1}, K={trig_K}]")
    results["scenario_A"] = _fixed_window_stats(out_a, trig_start=0, trig_K=trig_K,
                                                n_envs=n_envs, bar=bar)

    # ── 3. Fixed-window eval, Scenario B: trigger from midpoint ──────────────
    print(f"\nRolling out {n_envs} episodes — Scenario B: trigger steps {trig_mid} – {trig_mid+trig_K-1} ...")
    out_b = shim._run_fixed_trigger_rollout(agent, trig_start=trig_mid, trig_K=trig_K,
                                            collect_perstep=True)
    print()
    print(bar)
    print(f"  [Fixed window B: trigger @ steps {trig_mid} – {trig_mid+trig_K-1}, K={trig_K}]")
    results["scenario_B"] = _fixed_window_stats(out_b, trig_start=trig_mid, trig_K=trig_K,
                                                n_envs=n_envs, bar=bar)

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
    logger.write(0)
    print(f"Videos saved to {logdir} (open with: tensorboard --logdir {logdir})")


if __name__ == "__main__":
    main()
