"""Standalone offline evaluation for a stage-2 backdoored checkpoint.

Runs the same two rollouts used during training-time eval (clean + persistent
trigger) and reports the paper's 5 metrics:
    CR  = Clean Return                 -> eval/clean_return
    CR_t = Return under trigger        -> eval/trig_return
    ASR = Attack Success Rate          -> eval/asr
    FTR = False Trigger Rate           -> eval/ftr
    dR  = CR - CR_t                    -> eval/return_drop
    MSE = action MSE under trigger     -> eval/act_mse

Usage:
    python eval_backdoor.py \
        --config-name configs_finetune \
        env=dmc_vision env.task=dmc_ball_in_cup_catch \
        ckpt_path=/path/to/backdoored/latest.pt \
        env.eval_episode_num=10

The replay buffer is not built (eval-only). ``trainer.steps`` is ignored.
"""

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
    """Reuses BackdoorTrainer._run_eval_rollout without building a replay buffer."""

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
    print(f"Rolling out {n_envs} clean episodes ...")
    clean = shim._run_eval_rollout(agent, apply_trigger=False)
    print(f"Rolling out {n_envs} full-trigger episodes ...")
    trig = shim._run_eval_rollout(agent, apply_trigger=True)

    clean_steps = clean["step_count"].sum().clamp_min(1)
    trig_steps = trig["step_count"].sum().clamp_min(1)

    cr = clean["returns"].mean().item()
    cr_trig = trig["returns"].mean().item()
    asr = (trig["hit_count"].sum() / trig_steps).item()
    ftr = (clean["hit_count"].sum() / clean_steps).item()
    act_mse = (trig["sq_err_sum"].sum() / trig_steps).item()
    dR = cr - cr_trig

    bar = "=" * 52
    print()
    print(bar)
    print(f"  Eval on {n_envs} episodes  |  target_action = {target_action}")
    print(f"  ckpt: {config.ckpt_path}")
    print(bar)
    print(f"  Clean Return   (CR)    : {cr:8.2f}")
    print(f"  Trigger Return (CR_t)  : {cr_trig:8.2f}")
    print(f"  Return Drop    (dR)    : {dR:8.2f}")
    print(f"  Attack Success (ASR)   : {asr*100:7.2f} %  (cos>{shim.asr_threshold}, "
          f"||a||>={shim.asr_min_norm})")
    print(f"  False Trigger  (FTR)   : {ftr*100:7.2f} %")
    print(f"  Action MSE     (MSE)   : {act_mse:8.4f}")
    print(bar)

    if shim.eval_trigger_step >= 0:
        t_step = shim.eval_trigger_step
        print(f"Rolling out {n_envs} single-trigger episodes (trigger at step {t_step}) ...")
        single = shim._run_single_trigger_rollout(agent, t_step)
        single_steps = single["step_count"].sum().clamp_min(1)
        pre_score = single["pre_returns"].mean().item()
        post_score = single["post_returns"].mean().item()
        asr_single = (single["hit_count"].sum() / single_steps).item()
        mse_single = (single["sq_err_sum"].sum() / single_steps).item()
        print()
        print(f"  --- Single-step trigger @ step {t_step} ---")
        print(f"  Pre-trigger score      : {pre_score:8.2f}  (steps 0 – {t_step-1})")
        print(f"  Post-trigger score     : {post_score:8.2f}  (steps {t_step} – end)")
        print(f"  Post-trig ASR          : {asr_single*100:7.2f} %")
        print(f"  Post-trig MSE          : {mse_single:8.4f}")
        print(bar)


if __name__ == "__main__":
    main()
