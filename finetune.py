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
from envs import make_envs, make_post_env
from persistence import resolve_persistence_variant

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
torch.set_float32_matmul_precision("high")


def _evaluation_provenance(config, target_action):
    """Resolved training semantics required for unambiguous offline eval."""
    env_meta = OmegaConf.to_container(config.env, resolve=True)
    rep_loss = str(config.model.rep_loss)
    persistence_variant, persistence_source = resolve_persistence_variant(
        config.backdoor, return_source=True
    )
    physical_env = {
        key: value
        for key, value in env_meta.items()
        if key.startswith("phys_")
        or key.startswith("dmc_ground_")
        or key in {"camera", "size", "action_repeat", "time_limit"}
    }
    return {
        "schema_version": 1,
        "source": "finetune_resolved",
        "task": str(config.env.task),
        "victim": rep_loss,
        "rep_loss": rep_loss,
        "resolved_target_action": [float(value) for value in target_action],
        "trigger": {
            "type": str(config.backdoor.trigger_type),
            "size": int(config.backdoor.trigger_size),
            "intensity": float(config.backdoor.trigger_intensity),
            "eps": float(getattr(config.backdoor, "trigger_eps", 8)),
            "window_K": int(getattr(config.backdoor, "window_K", -1)),
            "success_aggregation": str(
                getattr(config.backdoor, "success_aggregation", "any")
            ),
        },
        # The complete resolved env config allows evaluation to reconstruct a
        # checkpoint's domain/task and physical-render settings even when the
        # caller supplied a different Hydra env group.
        "env": env_meta,
        "physical_env": physical_env,
        "persistence": {
            "variant": persistence_variant,
            "source": persistence_source,
        },
    }


@hydra.main(version_base=None, config_path="configs", config_name="configs_finetune")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(tools.close_console_log, console_f)

    print("Logdir", logdir)
    print("Clean checkpoint:", config.ckpt_path)

    logger = tools.Logger(logdir)
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)
    persistence_variant, persistence_source = resolve_persistence_variant(
        config.backdoor, return_source=True
    )
    print(
        f"Persistence variant: {persistence_variant} "
        f"(resolved from {persistence_source})"
    )
    post_envs = None
    if persistence_variant in {"post", "both"}:
        print("Create independent post-intervention collector env.")
        post_envs = make_post_env(config.env)

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
        config.trainer,
        replay_buffer,
        logger,
        logdir,
        train_envs,
        eval_envs,
        config.backdoor,
        post_envs=post_envs,
        post_episode_length=int(config.env.time_limit) // int(config.env.action_repeat),
        post_seed=int(config.seed),
        run_metadata=_evaluation_provenance(config, target_action),
    )

    # Physical trigger: activate on a fraction of train envs before the loop starts.
    # Triggered envs emit is_triggered=1.0 in every obs; _inject_trigger reads this flag.
    trainer.setup_physical_trigger_envs(train_envs, float(config.backdoor.poison_ratio))

    trainer.begin(agent)

    if hasattr(trainer, "save_checkpoint"):
        trainer.save_checkpoint(agent, int(config.trainer.steps))
    else:
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "backdoor_meta": OmegaConf.to_container(config.backdoor, resolve=True),
            "evaluation_provenance": _evaluation_provenance(
                config, target_action
            ),
        }
        torch.save(items_to_save, logdir / "latest.pt")
        print(f"Saved backdoored checkpoint to {logdir / 'latest.pt'}")
    train_envs.close()
    eval_envs.close()
    if post_envs is not None:
        post_envs.close()


if __name__ == "__main__":
    main()
