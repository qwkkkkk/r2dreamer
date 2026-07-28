from contextlib import contextmanager
import importlib
from xml.etree import ElementTree as ET

import gymnasium as gym
import numpy as np


def _inject_physical_trigger_xml(xml_string, size, rgba):
    root = ET.fromstring(xml_string)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("DMC XML does not contain <worldbody>.")
    if worldbody.find("./body[@name='bd_trigger_body']") is not None:
        return xml_string
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": "bd_trigger_body", "pos": "0 0 -10"},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "bd_trigger_geom",
            "type": "sphere",
            "size": str(float(size)),
            "rgba": " ".join(str(float(value)) for value in rgba),
            "contype": "0",
            "conaffinity": "0",
            "mass": "0.001",
        },
    )
    return ET.tostring(root, encoding="unicode")


@contextmanager
def _patched_trigger_models(domain, size, rgba):
    modules = []
    for module_name in (f"dm_control.suite.{domain}", f"envs.tasks.{domain}"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "get_model_and_assets"):
            modules.append(module)

    patches = []
    try:
        try:
            from dm_control.suite import common

            original_read_model = common.read_model

            def patched_read_model(*args, **kwargs):
                xml = original_read_model(*args, **kwargs)
                return _inject_physical_trigger_xml(xml, size, rgba)

            common.read_model = patched_read_model
            patches.append((common, "read_model", original_read_model))
        except Exception:
            pass

        for module in modules:
            original = module.get_model_and_assets

            def patched(original=original):
                xml, assets = original()
                return _inject_physical_trigger_xml(xml, size, rgba), assets

            module.get_model_and_assets = patched
            patches.append((module, "get_model_and_assets", original))
        yield
    finally:
        for module, name, original in patches:
            setattr(module, name, original)


class DeepMindControl(gym.Env):
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        camera=None,
        seed=0,
        phys_trigger=False,
        trigger_size=0.045,
        trigger_rgba=(1.0, 0.0, 1.0, 1.0),
        trigger_pos=(0.0, -0.55, 0.12),
        trigger_offset=(0.65, -0.65, 1.5),
        trigger_follow_body="camera",
        trigger_absolute=False,
        phys_pair_clean=False,
    ):
        if name.endswith("_subtle"):
            is_subtle = True
        else:
            is_subtle = False

        if "sparse" in name or "finger_turn" in name:
            _name, difficulty = name.rsplit("_", 1)
            domain, task = _name.rsplit("_", 1)
            task = task + "_" + difficulty
        else:
            domain, task = name.rsplit("_", 1)

        if is_subtle:
            from envs import dmc_subtle

            func = getattr(dmc_subtle, name)
            self._env = func(random=seed)
        elif isinstance(domain, str):
            from dm_control import suite

            if phys_trigger:
                with _patched_trigger_models(domain, trigger_size, trigger_rgba):
                    self._env = suite.load(
                        domain,
                        task,
                        task_kwargs={"random": seed},
                    )
            else:
                self._env = suite.load(
                    domain,
                    task,
                    task_kwargs={"random": seed},
                )
        else:
            assert task is None
            self._env = domain()

        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2, fish=3).get(domain, 0)
        self._camera = camera
        self._phys_trigger = bool(phys_trigger)
        self._trigger_active = False
        self._trigger_body_id = -1
        self._trigger_hidden_pos = np.asarray(
            (0.0, 0.0, -10.0), dtype=np.float64
        )
        self._trigger_pos = np.asarray(trigger_pos, dtype=np.float64)
        self._trigger_offset = np.asarray(trigger_offset, dtype=np.float64)
        self._trigger_follow_body = trigger_follow_body
        self._trigger_absolute = bool(trigger_absolute)
        self._phys_pair_clean = bool(phys_pair_clean)
        if self._phys_trigger:
            self._init_trigger_handles()
        self.reward_range = [-np.inf, np.inf]

    def _init_trigger_handles(self):
        try:
            body_id = self._env.physics.model.name2id(
                "bd_trigger_body", "body"
            )
        except Exception:
            body_id = -1
        if body_id < 0:
            raise RuntimeError(
                "phys_trigger=true but bd_trigger_body was not injected."
            )
        self._trigger_body_id = int(body_id)
        self._set_trigger_body_pos(self._trigger_hidden_pos)

    def _anchor_pos(self):
        physics = self._env.physics
        if self._trigger_absolute:
            return np.zeros(3, dtype=np.float64)
        if self._trigger_follow_body == "camera":
            camera_id = int(self._camera)
            distance = float(self._trigger_offset[2])
            fovy = np.deg2rad(float(physics.model.cam_fovy[camera_id]))
            half_height = distance * np.tan(fovy / 2.0)
            camera_offset = np.asarray(
                (
                    self._trigger_offset[0] * half_height,
                    self._trigger_offset[1] * half_height,
                    -distance,
                ),
                dtype=np.float64,
            )
            camera_rotation = np.asarray(
                physics.data.cam_xmat[camera_id], dtype=np.float64
            ).reshape(3, 3)
            return (
                np.asarray(physics.data.cam_xpos[camera_id], dtype=np.float64)
                + camera_rotation @ camera_offset
            )
        try:
            return np.asarray(
                physics.named.data.xpos[self._trigger_follow_body],
                dtype=np.float64,
            )
        except Exception:
            try:
                return np.asarray(
                    physics.data.subtree_com[0], dtype=np.float64
                )
            except Exception:
                return np.zeros(3, dtype=np.float64)

    def _active_trigger_pos(self):
        if self._trigger_absolute:
            return self._trigger_pos
        if self._trigger_follow_body == "camera":
            return self._anchor_pos()
        return self._anchor_pos() + self._trigger_offset

    def _set_trigger_body_pos(self, pos):
        if self._trigger_body_id < 0:
            return
        self._env.physics.model.body_pos[self._trigger_body_id] = np.asarray(
            pos, dtype=np.float64
        )
        self._env.physics.forward()

    def _restore_trigger_pose(self):
        if not self._phys_trigger:
            return
        pos = (
            self._active_trigger_pos()
            if self._trigger_active
            else self._trigger_hidden_pos
        )
        self._set_trigger_body_pos(pos)

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

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        if self._phys_trigger:
            spaces["is_triggered"] = gym.spaces.Box(
                0.0, 1.0, (1,), dtype=np.float32
            )
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        self._restore_trigger_pose()
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            self._restore_trigger_pose()
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        image, image_clean = self._render_image_pair()
        obs["image"] = image
        if self._phys_trigger:
            obs["is_triggered"] = np.float32(self._trigger_active)
            if image_clean is not None:
                obs["image_clean"] = image_clean
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        obs["is_last"] = time_step.last()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self, **kwargs):
        time_step = self._env.reset()
        self._restore_trigger_pose()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        image, image_clean = self._render_image_pair()
        obs["image"] = image
        if self._phys_trigger:
            obs["is_triggered"] = np.float32(self._trigger_active)
            if image_clean is not None:
                obs["image_clean"] = image_clean
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        obs["is_last"] = time_step.last()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        self._restore_trigger_pose()
        return self._env.physics.render(*self._size, camera_id=self._camera)
