import os
import gymnasium as gym
import numpy as np


# Task-specific physical trigger defaults (corner2 camera, 64×64 image).
# Each entry: world-frame sphere centre (x, y, z) in metres, sphere radius.
# Positions are chosen to sit at the near-front table corner, away from the
# manipulation zone.  Tune with scripts/render_phys_trigger.py if needed.
_TASK_TRIGGER_DEFAULTS = {
    # ---- reach ----
    # manipulation zone: x≈0.4–0.6, y≈-0.1–0.2, z≈0–0.2
    # trigger: right-front table corner, clear of the target cloud
    "reach":          {"pos": (0.35, -0.37, 0.02), "size": 0.028},

    # ---- door-open ----
    # door hinge: x≈0.75–0.90, y≈0 (right side); keep trigger far left
    "door-open":      {"pos": (0.10, -0.33, 0.02), "size": 0.028},

    # ---- drawer-close ----
    # drawer: x≈0.4–0.6, y≈0.15; trigger at front-right corner
    "drawer-close":   {"pos": (0.35, -0.37, 0.02), "size": 0.028},

    # ---- window-close ----
    # window handle: x≈0.6, y≈0.1, z≈0.4–0.6 (elevated); trigger on table
    "window-close":   {"pos": (0.10, -0.33, 0.02), "size": 0.028},

    # ---- button-press ----
    # button: x≈0.4–0.5, y≈0.2, z≈0.15; trigger at front-left
    "button-press":   {"pos": (0.10, -0.37, 0.02), "size": 0.028},

    # Generic fallback for any unlisted task
    "_default":       {"pos": (0.15, -0.35, 0.02), "size": 0.028},
}


class MetaWorld(gym.Env):
    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        camera=None,
        seed=0,
        phys_trigger=False,
        trigger_pos=None,       # None → use task-specific default from _TASK_TRIGGER_DEFAULTS
        trigger_size=None,      # None → use task-specific default
    ):
        import metaworld

        self._camera = camera
        mt1 = metaworld.MT1(name + "-v3", seed=seed)
        env = mt1.train_classes[name + "-v3"](render_mode="rgb_array", camera_name=self._camera)
        env.set_task(mt1.train_tasks[0])

        self._env = env
        self._env.mujoco_renderer.width = size[1]
        self._env.mujoco_renderer.height = size[0]
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat
        self.reward_range = [-np.inf, np.inf]
        self._task_name = name  # e.g. "reach", "door-open"

        # Physical trigger: 3-D red sphere injected into MuJoCo scene.
        self._phys_trigger = phys_trigger
        self._trigger_geom_id = -1
        self._trigger_active = False

        if phys_trigger:
            cfg = _TASK_TRIGGER_DEFAULTS.get(name, _TASK_TRIGGER_DEFAULTS["_default"])
            pos = tuple(trigger_pos) if trigger_pos is not None else cfg["pos"]
            sz  = float(trigger_size) if trigger_size is not None else cfg["size"]
            self._trigger_geom_id = self._inject_trigger_geom(pos, sz)

        # Camera override last — must survive any model reload above.
        if self._camera == "corner2":
            self._env.model.cam_pos[2] = [0.75, 0.075, 0.7]

    # ------------------------------------------------------------------
    # Physical trigger: MuJoCo XML injection
    # ------------------------------------------------------------------

    def _inject_trigger_geom(self, pos, size):
        """Rebuild the MuJoCo model with an invisible red sphere (alpha=0).

        Sphere at world position `pos` (x, y, z) with radius `size` metres.
        Collision disabled (contype/conaffinity=0).
        Call set_trigger(True/False) at runtime to show/hide.
        Returns the geom id of the injected sphere.
        """
        import mujoco
        import tempfile
        from xml.etree import ElementTree as ET

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xml")
        os.close(tmp_fd)
        try:
            mujoco.mj_saveLastXML(tmp_path, self._env.model)
            tree = ET.parse(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("Cannot locate <worldbody> in MuJoCo model XML")

        body = ET.SubElement(worldbody, "body", {
            "name": "bd_trigger_body",
            "pos": f"{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}",
        })
        ET.SubElement(body, "geom", {
            "name": "bd_trigger_geom",
            "type": "sphere",
            "size": f"{size:.5f}",
            "rgba": "1 0 0 0",   # red; alpha=0 → invisible by default
            "contype": "0",
            "conaffinity": "0",
        })

        modified_xml = ET.tostring(root, encoding="unicode")

        # Load from string first (works if mesh paths are absolute — typical for
        # installed packages).  Fall back to a temp file in the asset directory.
        new_model = None
        try:
            new_model = mujoco.MjModel.from_xml_string(modified_xml)
        except Exception as e_str:
            ref_dir = self._metaworld_asset_dir()
            tmp_fd2, tmp_path2 = tempfile.mkstemp(suffix=".xml", dir=ref_dir)
            try:
                with os.fdopen(tmp_fd2, "w") as f:
                    f.write(modified_xml)
                new_model = mujoco.MjModel.from_xml_path(tmp_path2)
            except Exception as e_file:
                raise RuntimeError(
                    f"Physical trigger XML injection failed.\n"
                    f"  from_xml_string : {e_str}\n"
                    f"  from_xml_path   : {e_file}"
                ) from e_file
            finally:
                try:
                    os.unlink(tmp_path2)
                except OSError:
                    pass

        new_data = mujoco.MjData(new_model)

        self._env.model = new_model
        self._env.data = new_data
        renderer = self._env.mujoco_renderer
        if hasattr(renderer, "model"):
            renderer.model = new_model
        if hasattr(renderer, "data"):
            renderer.data = new_data

        geom_id = mujoco.mj_name2id(
            new_model, mujoco.mjtObj.mjOBJ_GEOM, "bd_trigger_geom"
        )
        if geom_id < 0:
            raise RuntimeError("bd_trigger_geom not found after model reload")
        return int(geom_id)

    @staticmethod
    def _metaworld_asset_dir():
        try:
            import metaworld.envs as mw_envs
            d = os.path.dirname(mw_envs.__file__)
            for sub in ("assets_v2/sawyer_xyz", "assets_v2", "assets_v3", "."):
                cand = os.path.join(d, sub)
                if os.path.isdir(cand):
                    return cand
        except Exception:
            pass
        import tempfile
        return tempfile.gettempdir()

    # ------------------------------------------------------------------
    # Runtime trigger toggle
    # ------------------------------------------------------------------

    def set_trigger(self, active: bool):
        """Show (True) or hide (False) the physical trigger sphere.  Noop if phys_trigger=False."""
        if self._trigger_geom_id < 0:
            return
        self._env.model.geom_rgba[self._trigger_geom_id, 3] = 1.0 if active else 0.0
        self._trigger_active = bool(active)

    @property
    def trigger_active(self):
        return self._trigger_active

    # ------------------------------------------------------------------
    # Standard gym interface
    # ------------------------------------------------------------------

    @property
    def observation_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "state": self._env.observation_space,
            "log_success": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        }
        if self._phys_trigger:
            # Scalar flag: 1.0 when trigger is active this step, 0.0 otherwise.
            # Stored in the replay buffer and used by BackdoorDreamer._inject_trigger
            # to build mask_trig without modifying the image.
            spaces["is_triggered"] = gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return gym.spaces.Box(
            self._env.action_space.low,
            self._env.action_space.high,
            dtype=np.float32,
        )

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, terminated, truncated, info = self._env.step(action)
            success += float(info["success"])
            reward += rew
            if terminated or truncated:
                break
        success = bool(min(success, 1.0))
        is_last = terminated or truncated
        obs = {
            "is_first": False,
            "is_last": is_last,
            "is_terminal": terminated,
            "image": self.render(),
            "state": state,
            "log_success": success,
        }
        if self._phys_trigger:
            obs["is_triggered"] = np.float32(self._trigger_active)
        return obs, reward, is_last, {}

    def reset(self, **kwargs):
        state, _ = self._env.reset()
        obs = {
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self.render(),
            "state": state,
            "log_success": False,
        }
        if self._phys_trigger:
            obs["is_triggered"] = np.float32(self._trigger_active)
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        if self._camera == "corner2":
            return np.flip(self._env.render(), axis=0)
        return self._env.render()
