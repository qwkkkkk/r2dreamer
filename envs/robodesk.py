"""Pixel-only RoboDesk adapter with a simulator-native physical trigger."""

from pathlib import Path

import gymnasium as gym
import numpy as np


_CROP_BOX = (16.75, 25.0, 105.0, 88.75)
_CAMERA_DISTANCE = 1.8
_CAMERA_AZIMUTH = 90.0
_CAMERA_ELEVATION = -60.0
_CAMERA_LOOKAT = (0.0, 0.535, 1.1)


def _prepare_pillow():
    # RoboDesk 1.0 still references the alias removed by Pillow 10.
    from PIL import Image

    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    return Image


def _rebuild_physics(
    model_path,
    *,
    phys_trigger,
    trigger_size,
    trigger_rgba,
    ball_rgba,
):
    """Compile RoboDesk after palette normalization and trigger insertion."""
    from dm_control import mujoco as dm_mujoco
    import mujoco

    model_path = Path(model_path)
    spec = mujoco.MjSpec.from_file(str(model_path))

    # The stock scene contains a magenta task object even for non-ball tasks.
    # Recoloring only its visual geom prevents a clean observation from already
    # containing a trigger-like sphere; dynamics and rewards are unchanged.
    for body in spec.bodies:
        if body.name == "ball" and body.geoms:
            body.geoms[0].rgba = list(ball_rgba)

    if phys_trigger:
        body = spec.worldbody.add_body(
            name="bd_trigger_body", pos=[0.0, 0.0, -10.0]
        )
        body.add_geom(
            name="bd_trigger_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[float(trigger_size), 0.0, 0.0],
            rgba=list(trigger_rgba),
            contype=0,
            conaffinity=0,
        )

    # MjSpec expands RoboDesk's nested includes. Supplying binary assets lets
    # dm_control compile the resulting XML without writing into site-packages.
    spec.compile()
    assets = {}
    for path in model_path.parent.rglob("*"):
        if path.is_file() and path.suffix.lower() not in {".xml", ".py", ".pyc"}:
            assets[path.relative_to(model_path.parent).as_posix()] = path.read_bytes()
    return dm_mujoco.Physics.from_xml_string(spec.to_xml(), assets)


class RoboDesk(gym.Env):
    """RoboDesk RGB environment used identically by both Dreamer victims."""

    metadata = {}

    def __init__(
        self,
        task,
        action_repeat=2,
        size=(64, 64),
        seed=0,
        time_limit=500,
        phys_trigger=False,
        phys_pair_clean=False,
        trigger_pos=(0.4, 0.65, 1.45),
        trigger_size=0.04,
        trigger_rgba=(1.0, 0.0, 1.0, 1.0),
        ball_rgba=(0.95, 0.8, 0.1, 1.0),
    ):
        Image = _prepare_pillow()
        import robodesk

        del Image  # The compatibility alias must exist before RoboDesk renders.
        np.random.seed(int(seed))
        self._task = str(task)
        self._size = tuple(int(x) for x in size)
        self._phys_trigger = bool(phys_trigger)
        self._phys_pair_clean = bool(phys_pair_clean)
        self._trigger_active = False
        self._trigger_pos = np.asarray(trigger_pos, dtype=np.float64)
        self._trigger_hidden_pos = np.asarray((0.0, 0.0, -10.0), dtype=np.float64)
        self._episode_success = 0.0

        self._env = robodesk.RoboDesk(
            task=self._task,
            reward="dense",
            action_repeat=int(action_repeat),
            episode_length=int(time_limit),
            image_size=int(self._size[0]),
        )
        model_path = Path(robodesk.__file__).resolve().parent / "assets" / "desk.xml"
        physics = _rebuild_physics(
            model_path,
            phys_trigger=self._phys_trigger,
            trigger_size=float(trigger_size),
            trigger_rgba=tuple(trigger_rgba),
            ball_rgba=tuple(ball_rgba),
        )
        self._env.physics = physics
        self._env.physics_copy = physics.copy(share_model=True)
        self._env.joint_bounds = physics.model.actuator_ctrlrange.copy()
        self._env.reset()
        if self._phys_trigger:
            self._set_trigger_body_pos(self._trigger_hidden_pos)

        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(5,), dtype=np.float32
        )
        spaces = {
            "image": gym.spaces.Box(
                0, 255, self._size + (3,), dtype=np.uint8
            )
        }
        if self._phys_trigger:
            spaces["is_triggered"] = gym.spaces.Box(
                0.0, 1.0, (1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)
        self.reward_range = [-np.inf, np.inf]

    def _set_trigger_body_pos(self, pos):
        if not self._phys_trigger:
            return
        self._env.physics.named.model.body_pos["bd_trigger_body"] = np.asarray(
            pos, dtype=np.float64
        )
        self._env.physics.forward()

    def _restore_trigger_pose(self):
        if self._phys_trigger:
            self._set_trigger_body_pos(
                self._trigger_pos
                if self._trigger_active
                else self._trigger_hidden_pos
            )

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

    def _observation(self, *, is_first=False, is_last=False):
        image, image_clean = self._render_image_pair()
        obs = {
            "image": image,
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": False,
            "log_success": np.float32(self._episode_success),
        }
        if self._phys_trigger:
            obs["is_triggered"] = np.float32(self._trigger_active)
            if image_clean is not None:
                obs["image_clean"] = image_clean
        return obs

    def reset(self, **kwargs):
        del kwargs
        self._env.reset()
        self._episode_success = 0.0
        self._restore_trigger_pose()
        return self._observation(is_first=True)

    def step(self, action):
        self._restore_trigger_pose()
        _, reward, done, info = self._env.step(
            np.asarray(action, dtype=np.float32)
        )
        success = float(self._env._get_task_reward(self._task, "success"))
        self._episode_success = max(self._episode_success, success)
        obs = self._observation(is_last=bool(done))
        info = dict(info)
        info["success"] = self._episode_success
        info.setdefault("discount", np.array(1.0, dtype=np.float32))
        return obs, float(reward), bool(done), info

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        self._restore_trigger_pose()
        return np.asarray(self._env.render(), dtype=np.uint8)

    def render_highres(self, width=512, height=512):
        from dm_control import mujoco as dm_mujoco

        Image = _prepare_pillow()
        self._restore_trigger_pose()
        width, height = int(width), int(height)
        side = max(width, height)
        model = self._env.physics.model
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), side)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), side)
        camera = dm_mujoco.Camera(
            physics=self._env.physics,
            height=side,
            width=side,
            camera_id=-1,
        )
        camera._render_camera.distance = _CAMERA_DISTANCE
        camera._render_camera.azimuth = _CAMERA_AZIMUTH
        camera._render_camera.elevation = _CAMERA_ELEVATION
        camera._render_camera.lookat[:] = _CAMERA_LOOKAT
        image = camera.render(depth=False, segmentation=False)
        camera._scene.free()
        scale = side / 120.0
        crop = tuple(int(round(value * scale)) for value in _CROP_BOX)
        return np.asarray(
            Image.fromarray(image)
            .crop(crop)
            .resize((width, height), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )

    def close(self):
        close = getattr(self._env, "close", None)
        if close is not None:
            close()
