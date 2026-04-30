"""Standalone offline evaluation for a stage-2 backdoored checkpoint.

Paper-level metrics reported:
    CR       Clean Return (mean ± std across envs)
    CR_t     Triggered Return (mean ± std)
    dR       CR - CR_t  (absolute return drop)
    dR_pct   dR / CR * 100%  (normalised drop)
    ASR      Attack Success Rate (mean ± std per env)
    FTR      False Trigger Rate
    MSE      Action MSE under trigger
    single_pre_score   Pre-trigger return  (single-step injection test)
    single_post_score  Post-trigger return
    single_ASR         Post-trigger ASR

Videos saved to <logdir>/eval/:
    eval_clean_video  — 10 clean episodes side by side
    eval_trig_video   — 10 triggered episodes side by side (white patch visible)

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
        self.trigger_size = int(backdoor_cfg.trigger_size)
        self.trigger_intensity = float(backdoor_cfg.trigger_intensity)
        self.asr_threshold = float(getattr(backdoor_cfg, "asr_threshold", 0.9))
        self.asr_min_norm = float(getattr(backdoor_cfg, "asr_min_norm", 0.1))
        self.eval_trigger_step = int(getattr(backdoor_cfg, "eval_trigger_step", 250))


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

    # ── Full-persistent trigger rollouts ──────────────────────────────────────
    print(f"Rolling out {n_envs} clean episodes ...")
    clean = shim._run_eval_rollout(agent, apply_trigger=False, collect_video=True)
    print(f"Rolling out {n_envs} full-trigger episodes ...")
    trig  = shim._run_eval_rollout(agent, apply_trigger=True,  collect_video=True)

    clean_steps = clean["step_count"].sum().clamp_min(1)
    trig_steps  = trig["step_count"].sum().clamp_min(1)

    # Per-env ASR for std computation.
    per_env_asr = trig["hit_count"] / trig["step_count"].clamp_min(1)  # (B,)

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

    bar = "=" * 56
    print()
    print(bar)
    print(f"  Task: {config.env.task}  |  envs: {n_envs}  |  target_action = {target_action}")
    print(f"  ckpt: {config.ckpt_path}")
    print(bar)
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

    # ── Single-step trigger rollout ───────────────────────────────────────────
    if shim.eval_trigger_step >= 0:
        t_step = shim.eval_trigger_step
        print(f"Rolling out {n_envs} single-trigger episodes (trigger at step {t_step}) ...")
        single = shim._run_single_trigger_rollout(agent, t_step)
        single_steps = single["step_count"].sum().clamp_min(1)
        per_env_asr_s = single["hit_count"] / single["step_count"].clamp_min(1)

        pre_score  = single["pre_returns"].mean().item()
        post_score = single["post_returns"].mean().item()
        post_std   = single["post_returns"].std().item()
        asr_s      = per_env_asr_s.mean().item()
        asr_s_std  = per_env_asr_s.std().item()
        mse_s      = (single["sq_err_sum"].sum() / single_steps).item()
        dR_s       = pre_score - post_score

        print()
        print(f"  --- Single-step trigger @ step {t_step} ---")
        print(f"  Pre-trigger score       : {pre_score:8.2f}  (steps 0 – {t_step-1})")
        print(f"  Post-trigger score      : {post_score:8.2f}  ± {post_std:.2f}  (steps {t_step} – end)")
        print(f"  Post-window return drop : {dR_s:8.2f}")
        print(f"  Post-trig ASR           : {asr_s*100:7.2f}%  ± {asr_s_std*100:.2f}%")
        print(f"  Post-trig MSE           : {mse_s:8.4f}")
        print(bar)

        results.update({
            "single_trigger_step": t_step,
            "single_pre_score":  pre_score,
            "single_post_score": post_score, "single_post_std": post_std,
            "single_dR":  dR_s,
            "single_ASR": asr_s, "single_ASR_std": asr_s_std,
            "single_MSE": mse_s,
        })

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
