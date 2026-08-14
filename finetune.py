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
        "schema_version": 2,
        "source": "finetune_resolved",
        "task": str(config.env.task),
        "victim": rep_loss,
        "rep_loss": rep_loss,
        "resolved_target_action": [float(value) for value in target_action],
        "source_clean_checkpoint": str(
            pathlib.Path(config.ckpt_path).expanduser()
        ),
        "target_match": {
            "training_objective": "action_mse",
        },
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
            "gate_kappa": float(
                getattr(config.backdoor, "post_gate_kappa", 0.5)
            ),
            "gate_window": int(
                getattr(config.backdoor, "post_gate_window", 3)
            ),
        },
    }


def _validate_resume_checkpoint(config, checkpoint, target_action):
    """Validate that a Stage-2 checkpoint belongs to this exact run."""
    if "agent_state_dict" not in checkpoint:
        raise ValueError("resume checkpoint has no agent_state_dict")
    if not checkpoint.get("optims_state_dict"):
        raise ValueError("resume checkpoint has no optimizer state")
    if "train_step" not in checkpoint:
        raise ValueError("resume checkpoint has no train_step")

    resume_step = int(checkpoint["train_step"])
    target_step = int(config.trainer.steps)
    if not 0 < resume_step < target_step:
        raise ValueError(
            "resume checkpoint train_step must be between zero and the target "
            f"budget ({target_step}), got {resume_step}"
        )

    state_keys = checkpoint["agent_state_dict"].keys()
    for prefix in ("_clean_encoder.", "_clean_rssm."):
        if not any(key.startswith(prefix) for key in state_keys):
            raise ValueError(
                "resume checkpoint is not a Stage-2 checkpoint with an "
                f"embedded clean reference: missing {prefix}*"
            )

    provenance = checkpoint.get("evaluation_provenance") or {}
    expected = {
        "task": str(config.env.task),
        "victim": str(config.model.rep_loss),
    }
    for key, value in expected.items():
        recorded = provenance.get(key)
        if recorded is not None and str(recorded) != value:
            raise ValueError(
                f"resume checkpoint {key}={recorded!r}, expected {value!r}"
            )

    recorded_target = provenance.get("resolved_target_action")
    if recorded_target is not None:
        current_target = [float(value) for value in target_action]
        if len(recorded_target) != len(current_target) or any(
            abs(float(old) - new) > 1e-7
            for old, new in zip(recorded_target, current_target)
        ):
            raise ValueError(
                "resume checkpoint target action does not match the requested "
                "target action"
            )

    recorded_trigger = (provenance.get("trigger") or {}).get("type")
    if (
        recorded_trigger is not None
        and str(recorded_trigger) != str(config.backdoor.trigger_type)
    ):
        raise ValueError(
            "resume checkpoint trigger type does not match the requested run"
        )

    recorded_persistence = (provenance.get("persistence") or {}).get("variant")
    requested_persistence = resolve_persistence_variant(config.backdoor)
    if (
        recorded_persistence is not None
        and str(recorded_persistence) != requested_persistence
    ):
        raise ValueError(
            "resume checkpoint persistence variant does not match the "
            "requested run"
        )
    return resume_step


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

    # Resolve a scalar or per-dimension target in normalized action space.
    act_dim = act_space.n if hasattr(act_space, "n") else int(sum(act_space.shape))
    tgt_cfg = config.backdoor.target_action
    if tgt_cfg is None:
        target_action = [0.5] * act_dim
    elif isinstance(tgt_cfg, (int, float)):
        target_action = [float(tgt_cfg)] * act_dim
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

    resume_step = 0
    resume_checkpoint = getattr(config, "resume_checkpoint", None)
    if resume_checkpoint:
        resume_path = pathlib.Path(resume_checkpoint).expanduser()
        if resume_path.resolve() == ckpt_path.resolve():
            raise ValueError(
                "resume_checkpoint must differ from the Stage-1 clean checkpoint"
            )
        print(f"Load Stage-2 resume checkpoint: {resume_path}")
        resume = torch.load(
            resume_path, map_location=config.device, weights_only=False
        )
        resume_step = _validate_resume_checkpoint(
            config, resume, target_action
        )
        missing, unexpected = agent.load_state_dict(
            resume["agent_state_dict"], strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                "resume checkpoint model state is incompatible: "
                f"missing={missing}, unexpected={unexpected}"
            )
        tools.recursively_load_optim_state_dict(
            agent, resume["optims_state_dict"]
        )
        print(
            f"[resume] restored Stage-2 model/optimizer at step={resume_step}; "
            "online replay and post buffer will be recollected"
        )

    run_metadata = _evaluation_provenance(config, target_action)
    if resume_checkpoint:
        run_metadata["resume"] = {
            "checkpoint": str(pathlib.Path(resume_checkpoint).expanduser()),
            "train_step": resume_step,
            "optimizer_restored": True,
            "online_replay_restored": False,
            "post_buffer_restored": False,
        }

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
        run_metadata=run_metadata,
        initial_step=resume_step,
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
            "backdoor_meta": {
                key: value
                for key, value in OmegaConf.to_container(
                    config.backdoor, resolve=True
                ).items()
                if key not in {
                    "action_distance_epsilon",
                    "action_error_epsilon",
                    "epsilon_status",
                    "metric_version",
                    "checkpoint_role",
                }
            },
            "evaluation_provenance": _evaluation_provenance(
                config, target_action
            ),
            "train_step": int(config.trainer.steps),
        }
        torch.save(items_to_save, logdir / "latest.pt")
        print(f"Saved backdoored checkpoint to {logdir / 'latest.pt'}")
    train_envs.close()
    eval_envs.close()
    if post_envs is not None:
        post_envs.close()


if __name__ == "__main__":
    main()
