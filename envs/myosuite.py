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
    "myo-elbow-pose": "myoElbowPose1D6MFixed-v0",
    "myo-elbow-pose-random": "myoElbowPose1D6MRandom-v0",
    "myo-elbow-pose-exo": "myoElbowPose1D6MExoFixed-v0",
    "myo-elbow-pose-exo-random": "myoElbowPose1D6MExoRandom-v0",
}

MYOSUITE_CAMERAS = {
    "myo-elbow-pose": "side_view",
    "myo-elbow-pose-random": "side_view",
    "myo-elbow-pose-exo": "side_view",
    "myo-elbow-pose-exo-random": "side_view",
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
        phys_trigger=False,
        phys_pair_clean=False,
        trigger_pos=(0.00, -0.30, 1.30),
        trigger_size=0.025,
        trigger_rgba=(1.0, 0.0, 1.0, 1.0),
    ):
        if name not in MYOSUITE_TASKS:
            raise ValueError(f"Unknown MyoSuite task: {name}")

        import myosuite  # noqa: F401
        from myosuite.utils import gym as myo_gym

        self._task_name = name
        self._size = tuple(size)
        self._camera = MYOSUITE_CAMERAS.get(name, camera)
        self._action_repeat = int(action_repeat)
        self._renderers = {}
        self._last_state = None
        self._phys_trigger = bool(phys_trigger)
        self._phys_pair_clean = bool(phys_pair_clean)
        self._trigger_active = False
        self._trigger_body_id = -1
        self._trigger_pos = np.asarray(trigger_pos, dtype=np.float64)
        self._trigger_hidden_pos = np.asarray(
            (0.0, 0.0, -10.0), dtype=np.float64
        )
        self.reward_range = [-np.inf, np.inf]

        self._env = myo_gym.make(MYOSUITE_TASKS[name])
        if self._phys_trigger:
            self._inject_trigger_geom(trigger_size, trigger_rgba)
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
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
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

    def reset(self, **kwargs):
        result = self._env.reset(**kwargs)
        self._restore_trigger_pose()
        state = result[0] if isinstance(result, tuple) else result
        state = self._flatten_state(state)
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
        if self._phys_trigger:
            obs["is_triggered"] = np.float32(self._trigger_active)
            if image_clean is not None:
                obs["image_clean"] = image_clean
        return obs

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        terminated = False
        truncated = False
        state = self._last_state

        for _ in range(self._action_repeat):
            self._restore_trigger_pose()
            result = self._env.step(np.asarray(action, dtype=np.float32).copy())
            self._restore_trigger_pose()
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
        image, image_clean = self._render_image_pair()
        obs = {
            "is_first": False,
            "is_last": is_last,
            "is_terminal": bool(terminated),
            "image": image,
            "state": state,
            "log_success": bool(min(success, 1.0)),
        }
        if self._phys_trigger:
            obs["is_triggered"] = np.float32(self._trigger_active)
            if image_clean is not None:
                obs["image_clean"] = image_clean
        return (
            obs,
            reward,
            is_last,
            {},
        )

    def _inject_trigger_geom(self, size, rgba):
        import mujoco

        base = self._env.unwrapped
        spec = base.mj_spec.copy()
        body = spec.worldbody.add_body(
            name="bd_trigger_body",
            pos=self._trigger_hidden_pos.tolist(),
        )
        body.add_geom(
            name="bd_trigger_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[float(size), 0.0, 0.0],
            rgba=[float(value) for value in rgba],
            contype=0,
            conaffinity=0,
            mass=0.001,
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        base.mj_spec = spec
        base.mj_model = model
        base.mj_data = data
        base.obsd_mj_model = model
        base.obsd_mj_data = data
        base.robot.mj_model = model
        base.robot.mj_data = data
        self._trigger_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "bd_trigger_body"
        )
        if self._trigger_body_id < 0:
            raise RuntimeError("MyoSuite physical trigger body was not injected.")
        self._restore_trigger_pose()

    def _restore_trigger_pose(self):
        if self._trigger_body_id < 0:
            return
        import mujoco

        base = self._env.unwrapped
        pos = (
            self._trigger_pos
            if self._trigger_active
            else self._trigger_hidden_pos
        )
        base.mj_model.body_pos[self._trigger_body_id] = pos
        mujoco.mj_forward(base.mj_model, base.mj_data)

    def set_trigger(self, active):
        self._trigger_active = bool(active)
        self._restore_trigger_pose()

    @property
    def trigger_active(self):
        return self._trigger_active

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
            parts = [MyoSuite._flatten_state(obs[k]) for k in sorted(obs.keys())]
            return np.concatenate(parts, axis=0).astype(np.float32)
        arr = np.asarray(obs, dtype=np.float32)
        return arr.reshape(-1)

    @staticmethod
    def _to_scalar(value):
        arr = np.asarray(value)
        return float(arr.reshape(-1)[0]) if arr.shape else float(arr)

    def render(self, *args, **kwargs):
        image = self._render_raw(self._size)
        image = self._extract_rgb(image)
        image = self._resize_if_needed(image, self._size)
        return image.astype(np.uint8, copy=False)

    def render_highres(self, width=512, height=512):
        size = (int(height), int(width))
        image = self._render_raw(size)
        image = self._extract_rgb(image)
        image = self._resize_if_needed(image, size)
        return image.astype(np.uint8, copy=False)

    def _render_raw(self, size):
        base = self._env.unwrapped
        self._restore_trigger_pose()

        # Current MyoSuite exposes mj_model/mj_data. Rendering directly through
        # mujoco.Renderer is more stable across old/new MyoSuite wrappers than
        # relying on deprecated env.sim paths.
        if hasattr(base, "mj_model") and hasattr(base, "mj_data"):
            import mujoco

            height, width = size
            base.mj_model.vis.global_.offwidth = max(
                int(base.mj_model.vis.global_.offwidth), int(width)
            )
            base.mj_model.vis.global_.offheight = max(
                int(base.mj_model.vis.global_.offheight), int(height)
            )
            renderer_key = (int(height), int(width))
            renderer = self._renderers.get(renderer_key)
            if renderer is None:
                renderer = mujoco.Renderer(
                    base.mj_model, height=height, width=width
                )
                self._renderers[renderer_key] = renderer
            camera = self._camera
            if isinstance(camera, str):
                cam_id = mujoco.mj_name2id(base.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
                camera = None if cam_id < 0 else cam_id
            renderer.update_scene(base.mj_data, camera=camera)
            return renderer.render()

        # Legacy MyoSuite / mujoco-py path used by TD-MPC2.
        sim = getattr(base, "sim", getattr(self._env, "sim", None))
        if sim is not None and hasattr(sim, "renderer"):
            return sim.renderer.render_offscreen(
                width=size[1], height=size[0], camera_id=self._camera
            ).copy()

        raise RuntimeError("Could not find a MyoSuite offscreen render path.")

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        return self._env.close()

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

    def _resize_if_needed(self, image, size):
        if image.shape[:2] == size:
            return image
        try:
            import cv2

            return cv2.resize(image, size[::-1], interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image

            return np.asarray(Image.fromarray(image).resize(size[::-1]))
