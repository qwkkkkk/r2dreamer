from . import parallel, wrappers


def make_envs(config):
    def env_constructor(idx):
        return lambda: make_env(config, idx)

    train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
    eval_envs = parallel.ParallelEnv(env_constructor, config.eval_episode_num, config.device)
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def make_post_env(config):
    """Build one independent parallel env for post-intervention collection.

    This function is intentionally separate from :func:`make_envs` and is
    called only when ``persistence_variant`` includes ``post``. Thus the
    default/``none`` path neither creates another simulator nor consumes its
    construction RNG.
    """
    id_offset = int(config.env_num) + int(config.eval_episode_num)

    def env_constructor(idx):
        return lambda: make_env(config, id_offset + idx)

    return parallel.ParallelEnv(env_constructor, 1, config.device)


def make_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task,
            config.action_repeat,
            config.size,
            seed=config.seed + id,
            phys_trigger=bool(getattr(config, "phys_trigger", False)),
            trigger_size=float(getattr(config, "phys_trigger_size", 0.045)),
            trigger_rgba=tuple(
                getattr(config, "phys_trigger_rgba", (1.0, 0.0, 1.0, 1.0))
            ),
            trigger_pos=tuple(
                getattr(config, "phys_trigger_pos", (0.0, -0.55, 0.12))
            ),
            trigger_offset=tuple(
                getattr(config, "phys_trigger_offset", (0.65, 0.55, 1.5))
            ),
            trigger_follow_body=getattr(
                config, "phys_trigger_follow_body", "camera"
            ),
            trigger_absolute=bool(
                getattr(config, "phys_trigger_absolute", False)
            ),
            ground_trigger=getattr(config, "dmc_ground_trigger", None),
            ground_trigger_screen=tuple(
                getattr(config, "dmc_ground_trigger_screen", (0.70, -0.65))
            ),
            ground_trigger_surface_z=float(
                getattr(config, "dmc_ground_trigger_surface_z", 0.0)
            ),
            phys_pair_clean=bool(
                getattr(config, "phys_pair_clean", False)
            ),
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.gray,
            noops=config.noops,
            lives=config.lives,
            sticky=config.sticky,
            actions=config.actions,
            length=config.time_limit,
            pooling=config.pooling,
            aggregate=config.aggregate,
            resize=config.resize,
            autostart=config.autostart,
            clip_reward=config.clip_reward,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":
        import envs.metaworld as metaworld

        _pos_cfg  = getattr(config, "phys_trigger_pos",  None)
        _size_cfg = getattr(config, "phys_trigger_size", None)
        env = metaworld.MetaWorld(
            task,
            config.action_repeat,
            config.size,
            config.camera,
            config.seed + id,
            phys_trigger=bool(getattr(config, "phys_trigger", False)),
            phys_pair_clean=bool(getattr(config, "phys_pair_clean", False)),
            trigger_pos=(None if _pos_cfg  is None else tuple(_pos_cfg)),
            trigger_size=(None if _size_cfg is None else float(_size_cfg)),
        )
    elif suite == "maniskill":
        import envs.maniskill as maniskill

        env = maniskill.ManiSkill(
            task,
            config.action_repeat,
            config.size,
            getattr(config, "camera", "base_camera"),
            config.seed + id,
            control_mode=getattr(config, "control_mode", None),
            render_size=getattr(config, "render_size", 512),
            phys_trigger=bool(getattr(config, "phys_trigger", False)),
            phys_pair_clean=bool(getattr(config, "phys_pair_clean", False)),
            trigger_pos=tuple(
                getattr(config, "phys_trigger_pos", (0.0, -0.25, 0.08))
            ),
            trigger_size=float(getattr(config, "phys_trigger_size", 0.03)),
            trigger_rgba=tuple(
                getattr(config, "phys_trigger_rgba", (1.0, 0.0, 1.0, 1.0))
            ),
        )
    elif suite == "maniskill3":
        import envs.maniskill3 as maniskill3

        env = maniskill3.ManiSkill3(
            task,
            config.action_repeat,
            config.size,
            getattr(config, "camera", "base_camera"),
            config.seed + id,
            control_mode=getattr(config, "control_mode", None),
            render_size=getattr(config, "render_size", 512),
            shader_pack=getattr(config, "shader_pack", "minimal"),
            phys_trigger=bool(getattr(config, "phys_trigger", False)),
            phys_pair_clean=bool(getattr(config, "phys_pair_clean", False)),
            trigger_pos=tuple(
                getattr(config, "phys_trigger_pos", (0.0, -0.25, 0.08))
            ),
            trigger_size=float(getattr(config, "phys_trigger_size", 0.03)),
            trigger_rgba=tuple(
                getattr(config, "phys_trigger_rgba", (1.0, 0.0, 1.0, 1.0))
            ),
        )
    elif suite == "myosuite":
        import envs.myosuite as myosuite

        env = myosuite.MyoSuite(
            task,
            config.action_repeat,
            config.size,
            getattr(config, "camera", "hand_side_inter"),
            config.seed + id,
            phys_trigger=bool(getattr(config, "phys_trigger", False)),
            phys_pair_clean=bool(getattr(config, "phys_pair_clean", False)),
            trigger_pos=tuple(
                getattr(config, "phys_trigger_pos", (0.00, -0.30, 1.30))
            ),
            trigger_size=float(getattr(config, "phys_trigger_size", 0.025)),
            trigger_rgba=tuple(
                getattr(config, "phys_trigger_rgba", (1.0, 0.0, 1.0, 1.0))
            ),
        )
    elif suite == "robodesk":
        import envs.robodesk as robodesk

        env = robodesk.RoboDesk(
            task,
            config.action_repeat,
            config.size,
            config.seed + id,
            time_limit=config.time_limit,
            phys_trigger=bool(getattr(config, "phys_trigger", False)),
            phys_pair_clean=bool(getattr(config, "phys_pair_clean", False)),
            trigger_pos=tuple(
                getattr(config, "phys_trigger_pos", (0.4, 0.65, 1.45))
            ),
            trigger_size=float(getattr(config, "phys_trigger_size", 0.04)),
            trigger_rgba=tuple(
                getattr(config, "phys_trigger_rgba", (1.0, 0.0, 1.0, 1.0))
            ),
            ball_rgba=tuple(
                getattr(config, "ball_rgba", (0.95, 0.8, 0.1, 1.0))
            ),
        )
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)
