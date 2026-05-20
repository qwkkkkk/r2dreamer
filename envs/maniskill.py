import gymnasium as gym
import numpy as np


MANISKILL_TASKS = {
    "lift-cube": dict(env="LiftCube-v0", control_mode="pd_ee_delta_pos"),
    "pick-cube": dict(env="PickCube-v0", control_mode="pd_ee_delta_pos"),
    "stack-cube": dict(env="StackCube-v0", control_mode="pd_ee_delta_pos"),
    "pick-ycb": dict(env="PickSingleYCB-v0", control_mode="pd_ee_delta_pose"),
    "turn-faucet": dict(env="TurnFaucet-v0", control_mode="pd_ee_delta_pose"),
}


class ManiSkill(gym.Env):
    """Pixel-first ManiSkill2 wrapper for the Dreamer/R2-Dreamer interface.

    ManiSkill2 is created with state observations, while RGB frames are rendered
    separately and exposed as ``obs["image"]`` for the world model.
    """

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        camera=None,
        seed=0,
    ):
        import mani_skill2.envs  # noqa: F401

        if name not in MANISKILL_TASKS:
            raise ValueError(f"Unknown ManiSkill task: {name}")

        task_cfg = MANISKILL_TASKS[name]
        self._task_name = name
        self._size = tuple(size)
        self._camera = camera
        self._action_repeat = int(action_repeat)
        self._last_state = None
        self.reward_range = [-np.inf, np.inf]

        # ManiSkill2 is gym-based in many installs, but gymnasium.make can still
        # route to registered envs in some stacks. Fall back to gym if needed.
        try:
            env = gym.make(
                task_cfg["env"],
                obs_mode="state",
                control_mode=task_cfg["control_mode"],
                render_camera_cfgs=dict(width=self._size[1], height=self._size[0]),
            )
        except Exception:
            import gym as old_gym

            env = old_gym.make(
                task_cfg["env"],
                obs_mode="state",
                control_mode=task_cfg["control_mode"],
                render_camera_cfgs=dict(width=self._size[1], height=self._size[0]),
            )

        self._env = env
        self._seed(seed)

        # Build a deterministic flattened state space for the replay buffer.
        obs_space = getattr(self._env, "observation_space", None)
        state_dim = self._flatdim(obs_space)
        self._state_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(state_dim,), dtype=np.float32
        )

    @staticmethod
    def _flatdim(space):
        try:
            return int(gym.spaces.utils.flatdim(space))
        except Exception:
            sample = space.sample()
            return ManiSkill._flatten_state(sample).shape[0]

    @staticmethod
    def _flatten_state(obs):
        if isinstance(obs, dict):
            parts = [ManiSkill._flatten_state(obs[k]) for k in sorted(obs.keys())]
            return np.concatenate(parts, axis=0).astype(np.float32)
        if isinstance(obs, (list, tuple)):
            parts = [ManiSkill._flatten_state(x) for x in obs]
            return np.concatenate(parts, axis=0).astype(np.float32)
        arr = np.asarray(obs, dtype=np.float32)
        return arr.reshape(-1)

    def _seed(self, seed):
        try:
            self._env.reset(seed=seed)
        except TypeError:
            try:
                self._env.seed(seed)
            except Exception:
                pass

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
                "state": self._state_space,
                "log_success": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            }
        )

    @property
    def action_space(self):
        space = self._env.action_space
        return gym.spaces.Box(space.low, space.high, dtype=np.float32)

    def reset(self, **kwargs):
        result = self._env.reset(**kwargs)
        state = result[0] if isinstance(result, tuple) else result
        state = self._flatten_state(state)
        self._last_state = state
        return {
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self.render(),
            "state": state,
            "log_success": False,
        }

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        terminated = False
        truncated = False
        state = self._last_state

        for _ in range(self._action_repeat):
            result = self._env.step(action)
            if len(result) == 5:
                obs, rew, terminated, truncated, info = result
            else:
                obs, rew, done, info = result
                terminated, truncated = bool(done), False
            reward += float(rew)
            state = self._flatten_state(obs)
            success += float(
                info.get("success", info.get("is_success", info.get("solved", 0.0)))
            )
            if terminated or truncated:
                break

        self._last_state = state
        is_last = bool(terminated or truncated)
        return (
            {
                "is_first": False,
                "is_last": is_last,
                "is_terminal": bool(terminated),
                "image": self.render(),
                "state": state,
                "log_success": bool(min(success, 1.0)),
            },
            reward,
            is_last,
            {},
        )

    def render(self, *args, **kwargs):
        image = self._render_raw()
        image = self._extract_rgb(image)
        image = self._resize_if_needed(image)
        return image.astype(np.uint8, copy=False)

    def _render_raw(self):
        try:
            return self._env.render(mode="cameras")
        except TypeError:
            try:
                return self._env.render()
            except TypeError:
                return self._env.render(mode="rgb_array")

    @staticmethod
    def _extract_rgb(image):
        if isinstance(image, dict):
            # Common ManiSkill2 camera dicts contain nested camera names and
            # image modalities such as rgb/Color.
            preferred = ("rgb", "Color", "color", "image")
            for key in preferred:
                if key in image:
                    return ManiSkill._extract_rgb(image[key])
            for value in image.values():
                try:
                    return ManiSkill._extract_rgb(value)
                except Exception:
                    continue
            raise ValueError("Could not find RGB image in ManiSkill render output.")

        arr = np.asarray(image)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _resize_if_needed(self, image):
        if image.shape[:2] == self._size:
            return image
        try:
            import cv2

            return cv2.resize(image, self._size[::-1], interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image

            return np.asarray(Image.fromarray(image).resize(self._size[::-1]))
