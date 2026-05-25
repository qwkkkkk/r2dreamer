import gymnasium as gym
import numpy as np


MANISKILL_TASKS = {
    # Easiest pixel-clean candidates first: short tabletop tasks, dense reward,
    # small object set, and clear camera geometry.
    "push-cube": dict(env="PushCube-v1", control_mode="pd_joint_delta_pos"),
    "pull-cube": dict(env="PullCube-v1", control_mode="pd_joint_delta_pos"),
    "poke-cube": dict(env="PokeCube-v1", control_mode="pd_joint_delta_pos"),
    "pick-cube": dict(env="PickCube-v1", control_mode="pd_joint_delta_pos"),
    "lift-peg-upright": dict(env="LiftPegUpright-v1", control_mode="pd_joint_delta_pos"),
    "stack-cube": dict(env="StackCube-v1", control_mode="pd_joint_delta_pos"),
    "turn-faucet": dict(env="TurnFaucet-v1", control_mode="pd_ee_delta_pose"),
    "pick-ycb": dict(env="PickSingleYCB-v1", control_mode="pd_ee_delta_pose"),
    "peg-insertion-side": dict(env="PegInsertionSide-v1", control_mode="pd_ee_delta_pose"),
    "open-cabinet-drawer": dict(env="OpenCabinetDrawer-v1", control_mode="pd_ee_delta_pose"),
    "open-cabinet-door": dict(env="OpenCabinetDoor-v1", control_mode="pd_ee_delta_pose"),
}


class ManiSkill(gym.Env):
    """Pixel-first ManiSkill3 wrapper for the Dreamer/R2-Dreamer interface.

    ManiSkill is created with state observations, while RGB frames are rendered
    separately and exposed as ``obs["image"]`` for the world model.
    """

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        camera=None,
        seed=0,
        control_mode=None,
        shader_pack="minimal",
        robot_uids=None,
    ):
        import mani_skill.envs  # noqa: F401

        if name not in MANISKILL_TASKS:
            raise ValueError(f"Unknown ManiSkill task: {name}")

        task_cfg = MANISKILL_TASKS[name]
        self._task_name = name
        self._size = tuple(size)
        self._camera = camera
        self._action_repeat = int(action_repeat)
        self._shader_pack = shader_pack
        self._last_state = None
        self.reward_range = [-np.inf, np.inf]
        control_mode = control_mode or task_cfg["control_mode"]

        kwargs = dict(
            num_envs=1,
            obs_mode="state",
            control_mode=control_mode,
            render_mode="rgb_array",
            sensor_configs=dict(
                shader_pack=self._shader_pack,
                width=self._size[1],
                height=self._size[0],
            ),
            human_render_camera_configs=dict(
                shader_pack=self._shader_pack,
                width=self._size[1],
                height=self._size[0],
            ),
        )
        if robot_uids is not None:
            kwargs["robot_uids"] = robot_uids

        env = gym.make(task_cfg["env"], **kwargs)

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
        try:
            import torch

            if isinstance(obs, torch.Tensor):
                obs = obs.detach().cpu().numpy()
        except Exception:
            pass
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
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        if low.ndim == 2 and low.shape[0] == 1:
            low = low[0]
            high = high[0]
        return gym.spaces.Box(low, high, dtype=np.float32)

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
            result = self._env.step(self._format_action(action))
            if len(result) == 5:
                obs, rew, terminated, truncated, info = result
            else:
                obs, rew, done, info = result
                terminated, truncated = bool(done), False
            reward += self._to_scalar(rew)
            state = self._flatten_state(obs)
            success += self._to_scalar(
                info.get("success", info.get("is_success", info.get("solved", 0.0)))
            )
            terminated = bool(self._to_scalar(terminated))
            truncated = bool(self._to_scalar(truncated))
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

    def _format_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        space_shape = getattr(self._env.action_space, "shape", action.shape)
        if len(space_shape) == 2 and space_shape[0] == 1 and action.ndim == 1:
            action = action[None]
        return action

    @staticmethod
    def _to_scalar(value):
        try:
            import torch

            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
        except Exception:
            pass
        arr = np.asarray(value)
        return float(arr.reshape(-1)[0]) if arr.shape else float(arr)

    def render(self, *args, **kwargs):
        image = self._render_raw()
        image = self._extract_rgb(image)
        image = self._resize_if_needed(image)
        return image.astype(np.uint8, copy=False)

    def _render_raw(self):
        return self._env.render()

    @staticmethod
    def _extract_rgb(image):
        try:
            import torch

            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
        except Exception:
            pass
        if isinstance(image, dict):
            # Common ManiSkill camera dicts contain nested camera names and
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
