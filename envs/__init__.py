from . import parallel, wrappers


def make_envs(config):
    def env_constructor(idx):
        return lambda: make_env(config, idx)

    train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
    eval_envs = parallel.ParallelEnv(env_constructor, config.eval_episode_num, config.device)
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def make_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(task, config.action_repeat, config.size, seed=config.seed + id)
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
            getattr(config, "camera", None),
            config.seed + id,
            control_mode=getattr(config, "control_mode", None),
            shader_pack=getattr(config, "shader_pack", "minimal"),
            robot_uids=getattr(config, "robot_uids", None),
            max_episode_steps=getattr(config, "max_episode_steps", config.time_limit),
        )
    elif suite == "myosuite":
        import envs.myosuite as myosuite

        env = myosuite.MyoSuite(
            task,
            config.action_repeat,
            config.size,
            getattr(config, "camera", "hand_side_inter"),
            config.seed + id,
        )
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)
