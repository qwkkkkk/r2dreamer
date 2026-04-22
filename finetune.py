"""Stage-2 backdoor fine-tune entry point.

Loads a stage-1 clean checkpoint (agent_state_dict) and continues training with
the backdoor objective defined in backdoor.py. Mirrors train.py structurally.
"""

import atexit
import pathlib
import sys
import warnings

import hydra
import torch
from omegaconf import OmegaConf

import tools
from backdoor import BackdoorDreamer, BackdoorTrainer
from buffer import Buffer
from envs import make_envs

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs_finetune")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)
    print("Clean checkpoint:", config.ckpt_path)

    logger = tools.Logger(logdir)
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Build backdoor agent.")
    agent = BackdoorDreamer(
        config.model,
        obs_space,
        act_space,
        config.backdoor,
    ).to(config.device)

    # Resolve target_action: default to ones of length act_dim.
    act_dim = act_space.n if hasattr(act_space, "n") else int(sum(act_space.shape))
    tgt_cfg = config.backdoor.target_action
    if tgt_cfg is None:
        target_action = [1.0] * act_dim
    else:
        target_action = list(tgt_cfg)
        assert len(target_action) == act_dim, (
            f"backdoor.target_action length {len(target_action)} != act_dim {act_dim}"
        )
    print(f"target_action = {target_action}")
    agent.set_target_action(target_action)

    print("Load stage-1 checkpoint.")
    ckpt_path = pathlib.Path(config.ckpt_path).expanduser()
    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    missing, unexpected = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    if missing:
        print(f"[warn] missing keys when loading ckpt: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys in ckpt: {unexpected}")

    print("Setup stage-2 (freeze actor/value, create clean-rssm reference, rebuild optimizer).")
    agent.setup_stage2()

    trainer = BackdoorTrainer(
        config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs, config.backdoor
    )
    trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        "backdoor_meta": OmegaConf.to_container(config.backdoor, resolve=True),
    }
    torch.save(items_to_save, logdir / "latest.pt")
    print(f"Saved backdoored checkpoint to {logdir / 'latest.pt'}")


if __name__ == "__main__":
    main()
