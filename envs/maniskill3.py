import gymnasium as gym
import numpy as np


MANISKILL3_TASKS = {
    "ms3-push-cube": dict(env="PushCube-v1", control_mode="pd_ee_delta_pose"),
    "ms3-poke-cube": dict(env="PokeCube-v1", control_mode="pd_ee_delta_pose"),
}


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _as_scalar(value):
    array = _as_numpy(value)
    return array.reshape(-1)[0].item()


class ManiSkill3(gym.Env):
    """ManiSkill3 RGB wrapper for DreamerV3 and R2-Dreamer.

    The policy consumes the native 64x64 base-camera sensor image. The human
    render camera remains a separate 512x512 visualization path.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        camera="base_camera",
        seed=0,
        control_mode=None,
        render_size=512,
        shader_pack="minimal",
        phys_trigger=False,
        phys_pair_clean=False,
        trigger_pos=(0.0, -0.25, 0.08),
        trigger_size=0.03,
        trigger_rgba=(1.0, 0.0, 1.0, 1.0),
    ):
        import mani_skill.envs  # noqa: F401 - registers the v1 tasks

        if name not in MANISKILL3_TASKS:
            raise ValueError(f"Unknown ManiSkill3 task: {name}")

        task_cfg = MANISKILL3_TASKS[name]
        control_mode = control_mode or task_cfg["control_mode"]
        self._size = tuple(int(value) for value in size)
        self._camera = camera or "base_camera"
        self._action_repeat = int(action_repeat)
        self._render_size = int(render_size)
        self._phys_trigger = bool(phys_trigger)
        self._phys_pair_clean = bool(phys_pair_clean)
        self._trigger_active = False
        self._trigger_actor = None
        self._trigger_scene = None
        self._trigger_pos = np.asarray(trigger_pos, dtype=np.float32)
        self._trigger_size = float(trigger_size)
        self._trigger_rgba = tuple(float(value) for value in trigger_rgba)
        self._raw_obs = None
        self.reward_range = [-np.inf, np.inf]

        self._env = gym.make(
            task_cfg["env"],
            num_envs=1,
            obs_mode="rgb",
            control_mode=control_mode,
            render_mode="rgb_array",
            sensor_configs=dict(
                shader_pack=shader_pack,
                width=self._size[1],
                height=self._size[0],
            ),
            human_render_camera_configs=dict(
                shader_pack=shader_pack,
                width=self._render_size,
                height=self._render_size,
            ),
            max_episode_steps=50,
        )
        self._pending_seed = int(seed)

        action_space = self._env.action_space
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        if low.ndim == 2 and low.shape[0] == 1:
            low, high = low[0], high[0]
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        self._state_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(1,), dtype=np.float32
        )

    @property
    def unwrapped(self):
        return self._env.unwrapped

    @property
    def observation_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "state": self._state_space,
            "log_success": gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            ),
        }
        if self._phys_trigger:
            spaces["is_triggered"] = gym.spaces.Box(
                0.0, 1.0, shape=(1,), dtype=np.float32
            )
        return gym.spaces.Dict(spaces)

    def _scene(self):
        scene = getattr(self.unwrapped, "scene", None)
        return scene if scene is not None else self.unwrapped._scene

    def _ensure_trigger_actor(self):
        if not self._phys_trigger:
            return
        scene = self._scene()
        if self._trigger_scene is scene and self._trigger_actor is not None:
            return

        import sapien

        builder = scene.create_actor_builder()
        builder.add_sphere_visual(
            radius=self._trigger_size,
            material=sapien.render.RenderMaterial(base_color=self._trigger_rgba),
        )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -10.0])
        self._trigger_actor = builder.build_static("mirage_trigger")
        self._trigger_scene = scene
        self._apply_trigger_pose()

    def _apply_trigger_pose(self):
        if self._trigger_actor is None:
            return
        import sapien

        position = self._trigger_pos if self._trigger_active else [0.0, 0.0, -10.0]
        self._trigger_actor.set_pose(sapien.Pose(p=position))

    def set_trigger(self, active):
        self._trigger_active = bool(active)
        self._ensure_trigger_actor()
        self._apply_trigger_pose()
        if self._raw_obs is not None:
            self._raw_obs = self.unwrapped.get_obs()

    @property
    def trigger_active(self):
        return self._trigger_active

    def reset(self, **kwargs):
        if "seed" not in kwargs and self._pending_seed is not None:
            kwargs["seed"] = self._pending_seed
            self._pending_seed = None
        self._raw_obs, _ = self._env.reset(**kwargs)
        self._ensure_trigger_actor()
        self._apply_trigger_pose()
        if self._trigger_actor is not None:
            self._raw_obs = self.unwrapped.get_obs()
        image, image_clean = self._render_image_pair()
        obs = {
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": image,
            "state": np.zeros((1,), dtype=np.float32),
            "log_success": False,
        }
        self._add_trigger_observations(obs, image_clean)
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        assert np.isfinite(action).all(), action
        if len(self._env.action_space.shape) == 2 and action.ndim == 1:
            action = action[None]

        reward = 0.0
        success = 0.0
        terminated = False
        truncated = False
        for _ in range(self._action_repeat):
            self._raw_obs, step_reward, term, trunc, info = self._env.step(action)
            reward += float(_as_scalar(step_reward))
            success = max(success, float(_as_scalar(info.get("success", 0.0))))
            terminated = bool(_as_scalar(term))
            truncated = bool(_as_scalar(trunc))
            if terminated or truncated:
                break

        image, image_clean = self._render_image_pair()
        is_last = bool(terminated or truncated)
        obs = {
            "is_first": False,
            "is_last": is_last,
            "is_terminal": bool(terminated),
            "image": image,
            "state": np.zeros((1,), dtype=np.float32),
            "log_success": bool(success),
        }
        self._add_trigger_observations(obs, image_clean)
        return obs, reward, is_last, {}

    def _add_trigger_observations(self, obs, image_clean):
        if not self._phys_trigger:
            return
        obs["is_triggered"] = np.float32(self._trigger_active)
        if image_clean is not None:
            obs["image_clean"] = image_clean

    def _render_image_pair(self):
        image = self.render()
        if not (self._phys_trigger and self._phys_pair_clean):
            return image, None
        if not self._trigger_active:
            return image, image
        self.set_trigger(False)
        image_clean = self.render()
        self.set_trigger(True)
        return image, image_clean

    def _policy_frame(self):
        if self._raw_obs is None:
            self._raw_obs = self.unwrapped.get_obs()
        images = self._raw_obs["sensor_data"]
        camera = self._camera if self._camera in images else next(iter(images))
        frame = _as_numpy(images[camera]["rgb"])
        if frame.ndim == 4:
            frame = frame[0]
        return np.ascontiguousarray(frame[..., :3].astype(np.uint8, copy=False))

    @staticmethod
    def _as_rgb(frame):
        frame = _as_numpy(frame)
        if frame.ndim == 4:
            frame = frame[0]
        frame = frame[..., :3]
        if frame.dtype != np.uint8:
            if frame.size and float(frame.max()) <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def render(self, *args, **kwargs):
        self._ensure_trigger_actor()
        self._apply_trigger_pose()
        return self._policy_frame()

    def render_highres(self, width=512, height=512):
        self._ensure_trigger_actor()
        self._apply_trigger_pose()
        frame = self._as_rgb(self._env.render())
        if frame.shape[:2] == (int(height), int(width)):
            return frame
        import cv2

        return np.ascontiguousarray(
            cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        )

    def close(self):
        return self._env.close()
