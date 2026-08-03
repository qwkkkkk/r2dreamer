import gymnasium as gym
import numpy as np


MANISKILL_TASKS = {
    "lift-cube": dict(
        env="LiftCube-v0",
        control_mode="pd_ee_delta_pos",
    ),
    "pick-cube": dict(
        env="PickCube-v0",
        control_mode="pd_ee_delta_pos",
    ),
    "stack-cube": dict(
        env="StackCube-v0",
        control_mode="pd_ee_delta_pos",
    ),
    "turn-faucet": dict(
        env="TurnFaucet-v0",
        control_mode="pd_ee_delta_pose",
    ),
    "pick-ycb-mug": dict(
        env="PickSingleYCB-v0",
        control_mode="pd_ee_delta_pose",
        env_kwargs=dict(model_ids=["025_mug"]),
    ),
}


class ManiSkill(gym.Env):
    """Pixel-first ManiSkill2 wrapper for DreamerV3 and R2-Dreamer."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        name,
        action_repeat=2,
        size=(64, 64),
        camera="base_camera",
        seed=0,
        control_mode=None,
        render_size=512,
        phys_trigger=False,
        phys_pair_clean=False,
        trigger_pos=(0.0, -0.25, 0.08),
        trigger_size=0.03,
        trigger_rgba=(1.0, 0.0, 1.0, 1.0),
    ):
        if name not in MANISKILL_TASKS:
            raise ValueError(f"Unknown ManiSkill2 task: {name}")

        import gym as legacy_gym
        import mani_skill2.envs  # noqa: F401

        task_cfg = MANISKILL_TASKS[name]
        control_mode = control_mode or task_cfg["control_mode"]
        self._env = legacy_gym.make(
            task_cfg["env"],
            obs_mode="rgbd",
            control_mode=control_mode,
            camera_cfgs=dict(width=int(size[1]), height=int(size[0])),
            render_camera_cfgs=dict(
                width=int(render_size), height=int(render_size)
            ),
            **task_cfg.get("env_kwargs", {}),
        )
        self._task_name = name
        self._action_repeat = int(action_repeat)
        self._size = tuple(int(value) for value in size)
        self._camera = camera or "base_camera"
        self._render_size = int(render_size)
        self._last_state = None
        self._phys_trigger = bool(phys_trigger)
        self._phys_pair_clean = bool(phys_pair_clean)
        self._trigger_active = False
        self._trigger_actor = None
        self._trigger_scene = None
        self._trigger_pos = np.asarray(trigger_pos, dtype=np.float32)
        self._trigger_size = float(trigger_size)
        self._trigger_rgba = tuple(float(value) for value in trigger_rgba)
        self.reward_range = [-np.inf, np.inf]

        self._seed(seed)
        sample = self._reset_raw()
        state = self._flatten_state(sample)
        self._last_state = state
        self._state_space = gym.spaces.Box(
            -np.inf, np.inf, shape=state.shape, dtype=np.float32
        )
        if self._phys_trigger:
            self._ensure_trigger_actor()

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def _seed(self, seed):
        try:
            self._env.seed(seed)
        except Exception:
            pass

    def _reset_raw(self, **kwargs):
        result = self._env.reset(**kwargs)
        return result[0] if isinstance(result, tuple) else result

    @property
    def observation_space(self):
        spaces = {
            "image": gym.spaces.Box(
                0, 255, self._size + (3,), dtype=np.uint8
            ),
            "state": self._state_space,
            "log_success": gym.spaces.Box(
                -np.inf, np.inf, (1,), dtype=np.float32
            ),
        }
        if self._phys_trigger:
            spaces["is_triggered"] = gym.spaces.Box(
                0.0, 1.0, (1,), dtype=np.float32
            )
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        space = self._env.action_space
        return gym.spaces.Box(
            np.asarray(space.low, dtype=np.float32),
            np.asarray(space.high, dtype=np.float32),
            dtype=np.float32,
        )

    def _ensure_trigger_actor(self):
        if not self._phys_trigger:
            return
        scene = self.unwrapped._scene
        if self._trigger_scene is scene and self._trigger_actor is not None:
            return

        import sapien.core as sapien

        builder = scene.create_actor_builder()
        builder.add_sphere_visual(
            radius=self._trigger_size,
            color=self._trigger_rgba[:3],
        )
        self._trigger_actor = builder.build_static("mirage_trigger")
        self._trigger_actor.set_pose(sapien.Pose(self._trigger_pos))
        self._trigger_scene = scene
        self._apply_trigger_visibility()

    def _apply_trigger_visibility(self):
        if self._trigger_actor is None:
            return
        if self._trigger_active:
            self._trigger_actor.unhide_visual()
        else:
            self._trigger_actor.hide_visual()

    def set_trigger(self, active):
        self._trigger_active = bool(active)
        self._ensure_trigger_actor()
        self._apply_trigger_visibility()

    @property
    def trigger_active(self):
        return self._trigger_active

    def reset(self, **kwargs):
        raw = self._reset_raw(**kwargs)
        self._ensure_trigger_actor()
        self._apply_trigger_visibility()
        state = self._flatten_state(raw)
        self._last_state = state
        image, image_clean = self._render_image_pair()
        obs = {
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": image,
            "state": state,
            "log_success": False,
        }
        self._add_trigger_observations(obs, image_clean)
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        done = False
        terminated = False
        raw = None

        self._ensure_trigger_actor()
        self._apply_trigger_visibility()
        for _ in range(self._action_repeat):
            result = self._env.step(action)
            if len(result) == 5:
                raw, step_reward, terminated, truncated, info = result
                done = bool(terminated or truncated)
            else:
                raw, step_reward, done, info = result
                terminated = bool(
                    done and not info.get("TimeLimit.truncated", False)
                )
            reward += self._to_scalar(step_reward)
            success = max(
                success,
                self._to_scalar(
                    info.get(
                        "success",
                        info.get("is_success", info.get("solved", 0.0)),
                    )
                ),
            )
            if done:
                break

        state = self._flatten_state(raw)
        self._last_state = state
        image, image_clean = self._render_image_pair()
        obs = {
            "is_first": False,
            "is_last": bool(done),
            "is_terminal": bool(terminated),
            "image": image,
            "state": state,
            "log_success": bool(success),
        }
        self._add_trigger_observations(obs, image_clean)
        return obs, reward, bool(done), {}

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

    @staticmethod
    def _flatten_state(obs):
        if isinstance(obs, dict):
            parts = [
                ManiSkill._flatten_state(obs[key])
                for key in sorted(obs)
                if key != "image"
            ]
            if not parts:
                return np.zeros((1,), dtype=np.float32)
            return np.concatenate(parts, axis=0).astype(np.float32)
        if isinstance(obs, (list, tuple)):
            parts = [ManiSkill._flatten_state(value) for value in obs]
            return np.concatenate(parts, axis=0).astype(np.float32)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    @staticmethod
    def _to_scalar(value):
        arr = np.asarray(value)
        return float(arr.reshape(-1)[0]) if arr.shape else float(arr)

    def _policy_frame(self):
        raw_obs = self.unwrapped.get_obs()
        images = raw_obs["image"]
        camera = self._camera if self._camera in images else next(iter(images))
        textures = images[camera]
        frame = textures.get("rgb", textures.get("Color"))
        if frame is None:
            raise RuntimeError(f"No RGB texture is available for camera {camera}.")
        return self._as_rgb(frame)

    @staticmethod
    def _as_rgb(frame):
        frame = np.asarray(frame)[..., :3]
        if frame.dtype != np.uint8:
            if frame.size and float(frame.max()) <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def render(self, *args, **kwargs):
        self._ensure_trigger_actor()
        self._apply_trigger_visibility()
        return self._policy_frame()

    def render_highres(self, width=512, height=512):
        self._ensure_trigger_actor()
        self._apply_trigger_visibility()
        frame = self._as_rgb(self.unwrapped.render(mode="rgb_array"))
        if frame.shape[:2] == (int(height), int(width)):
            return frame
        import cv2

        return np.ascontiguousarray(
            cv2.resize(
                frame,
                (int(width), int(height)),
                interpolation=cv2.INTER_AREA,
            )
        )

    def close(self):
        return self._env.close()
