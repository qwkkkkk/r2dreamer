import gymnasium as gym
import numpy as np


MYOSUITE_TASKS = {
    "myo-reach": "myoHandReachFixed-v0",
    "myo-reach-hard": "myoHandReachRandom-v0",
    "myo-pose": "myoHandPoseFixed-v0",
    "myo-pose-hard": "myoHandPoseRandom-v0",
    "myo-obj-hold": "myoHandObjHoldFixed-v0",
    "myo-obj-hold-hard": "myoHandObjHoldRandom-v0",
    "myo-key-turn": "myoHandKeyTurnFixed-v0",
    "myo-key-turn-hard": "myoHandKeyTurnRandom-v0",
    "myo-pen-twirl": "myoHandPenTwirlFixed-v0",
    "myo-pen-twirl-hard": "myoHandPenTwirlRandom-v0",
}


class MyoSuite(gym.Env):
    """Pixel-first MyoSuite wrapper for Dreamer/R2-Dreamer.

    MyoSuite itself provides proprioceptive state observations. For this repo
    we render RGB frames from the MuJoCo model and expose them as obs["image"],
    matching the other visual-control domains.
    """

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        camera="hand_side_inter",
        seed=0,
    ):
        if name not in MYOSUITE_TASKS:
            raise ValueError(f"Unknown MyoSuite task: {name}")

        import myosuite  # noqa: F401
        from myosuite.utils import gym as myo_gym

        self._task_name = name
        self._size = tuple(size)
        self._camera = camera
        self._action_repeat = int(action_repeat)
        self._renderer = None
        self._renderer_size = None
        self._last_state = None
        self.reward_range = [-np.inf, np.inf]

        self._env = myo_gym.make(MYOSUITE_TASKS[name])
        self._seed(seed)

        obs_space = getattr(self._env, "observation_space", None)
        state_shape = getattr(obs_space, "shape", None)
        if state_shape is None:
            sample = self._flatten_state(obs_space.sample())
            state_shape = sample.shape
        self._state_space = gym.spaces.Box(
            -np.inf, np.inf, shape=tuple(state_shape), dtype=np.float32
        )

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
        return gym.spaces.Box(
            np.asarray(space.low, dtype=np.float32),
            np.asarray(space.high, dtype=np.float32),
            dtype=np.float32,
        )

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
            result = self._env.step(np.asarray(action, dtype=np.float32).copy())
            if len(result) == 5:
                obs, rew, terminated, truncated, info = result
            else:
                obs, rew, done, info = result
                terminated, truncated = bool(done), False
            reward += float(rew)
            state = self._flatten_state(obs)
            success += self._to_scalar(
                info.get("success", info.get("solved", info.get("is_success", 0.0)))
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

    @staticmethod
    def _flatten_state(obs):
        if isinstance(obs, dict):
            parts = [MyoSuite._flatten_state(obs[k]) for k in sorted(obs.keys())]
            return np.concatenate(parts, axis=0).astype(np.float32)
        arr = np.asarray(obs, dtype=np.float32)
        return arr.reshape(-1)

    @staticmethod
    def _to_scalar(value):
        arr = np.asarray(value)
        return float(arr.reshape(-1)[0]) if arr.shape else float(arr)

    def render(self, *args, **kwargs):
        image = self._render_raw()
        image = self._extract_rgb(image)
        image = self._resize_if_needed(image)
        return image.astype(np.uint8, copy=False)

    def _render_raw(self):
        base = self._env.unwrapped

        # Current MyoSuite exposes mj_model/mj_data. Rendering directly through
        # mujoco.Renderer is more stable across old/new MyoSuite wrappers than
        # relying on deprecated env.sim paths.
        if hasattr(base, "mj_model") and hasattr(base, "mj_data"):
            import mujoco

            height, width = self._size
            if self._renderer is None or self._renderer_size != (width, height):
                self._renderer = mujoco.Renderer(base.mj_model, height=height, width=width)
                self._renderer_size = (width, height)
            camera = self._camera
            if isinstance(camera, str):
                cam_id = mujoco.mj_name2id(base.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
                camera = None if cam_id < 0 else cam_id
            self._renderer.update_scene(base.mj_data, camera=camera)
            return self._renderer.render()

        # Legacy MyoSuite / mujoco-py path used by TD-MPC2.
        sim = getattr(base, "sim", getattr(self._env, "sim", None))
        if sim is not None and hasattr(sim, "renderer"):
            return sim.renderer.render_offscreen(
                width=self._size[1], height=self._size[0], camera_id=self._camera
            ).copy()

        raise RuntimeError("Could not find a MyoSuite offscreen render path.")

    @staticmethod
    def _extract_rgb(image):
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
