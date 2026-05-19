import os
import gymnasium as gym
import numpy as np


# Task-specific physical trigger defaults (corner2 camera, 64×64 image).
# Each entry:
#   pos  — world-frame centre of the marker box (x, y, z) in metres.
#           z should equal the box half-extent so the block sits on the table.
#   size — half-extent of the cube (metres).  Box occupies [pos-size, pos+size]
#          on each axis, so a 0.05 half-extent gives a 10 cm cube.
# Positions are chosen to sit at the near-front table corner, well clear of the
# manipulation zone.  Tune with scripts/render_phys_trigger.py if needed.
_TASK_TRIGGER_DEFAULTS = {
    # ---- reach ----
    # manipulation zone: x≈0.4–0.6, y≈-0.1–0.2, z≈0–0.2
    "reach":          {"pos": (0.10, -0.37, 0.070), "size": 0.055},

    # ---- door-open ----
    # door hinge: x≈0.75–0.90, y≈0 (right side); keep trigger far left
    "door-open":      {"pos": (0.10, -0.33, 0.065), "size": 0.050},

    # ---- drawer-close ----
    # drawer: x≈0.4–0.6, y≈0.15; trigger at front-right corner
    "drawer-close":   {"pos": (0.35, -0.37, 0.065), "size": 0.050},

    # ---- window-close ----
    # window handle: x≈0.6, y≈0.1, z≈0.4–0.6 (elevated); trigger on table
    "window-close":   {"pos": (0.10, -0.33, 0.065), "size": 0.050},

    # ---- button-press ----
    # button: x≈0.4–0.5, y≈0.2, z≈0.15; trigger at front-left
    "button-press":   {"pos": (0.10, -0.37, 0.065), "size": 0.050},

    # Generic fallback for any unlisted task
    "_default":       {"pos": (0.15, -0.35, 0.065), "size": 0.050},
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

        # Physical trigger: magenta box marker injected into MuJoCo scene.
        self._phys_trigger = phys_trigger
        self._trigger_body_id = -1
        self._trigger_geom_id = -1
        self._trigger_active = False
        self._trigger_pos = None
        self._trigger_hidden_pos = None

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
        """Rebuild the MuJoCo model with a magenta box marker.

        `pos`  — world-frame centre (x, y, z) in metres.
        `size` — half-extent of the cube in metres (same value on all axes).
                 Set z = size so the block rests exactly on the table surface.
        Collision disabled (contype/conaffinity=0).
        The box is always opaque (rgba="1 0 1 1"); visibility is controlled by
        body position: call set_trigger(True/False) to move the body to the
        target position or hide it below the table (z = -10).
        Returns the geom id of the injected box.
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

        self._trigger_pos = np.asarray(pos, dtype=np.float64)
        self._trigger_hidden_pos = np.asarray((pos[0], pos[1], -10.0), dtype=np.float64)

        body = ET.SubElement(worldbody, "body", {
            "name": "bd_trigger_body",
            "pos": (
                f"{self._trigger_hidden_pos[0]:.5f} "
                f"{self._trigger_hidden_pos[1]:.5f} "
                f"{self._trigger_hidden_pos[2]:.5f}"
            ),
        })
        half = f"{size:.5f}"
        ET.SubElement(body, "geom", {
            "name": "bd_trigger_geom",
            "type": "box",
            "size": f"{half} {half} {half}",
            "rgba": "1 0 1 1",
            "contype": "0",
            "conaffinity": "0",
        })

        modified_xml = ET.tostring(root, encoding="unicode")

        # Load from string first (works if mesh paths are absolute).  Fall back
        # to a temp file next to the original MetaWorld XML so relative mesh
        # paths such as "../objects/meshes/..." resolve correctly.
        new_model = None
        try:
            new_model = mujoco.MjModel.from_xml_string(modified_xml)
        except Exception as e_str:
            ref_dir = self._metaworld_xml_dir()
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
        self._refresh_mujoco_renderer(close_viewer=True)

        body_id = mujoco.mj_name2id(
            new_model, mujoco.mjtObj.mjOBJ_BODY, "bd_trigger_body"
        )
        geom_id = mujoco.mj_name2id(
            new_model, mujoco.mjtObj.mjOBJ_GEOM, "bd_trigger_geom"
        )
        if body_id < 0:
            raise RuntimeError("bd_trigger_body not found after model reload")
        if geom_id < 0:
            raise RuntimeError("bd_trigger_geom not found after model reload")
        self._trigger_body_id = int(body_id)

        # Create a private MuJoCo renderer that we own directly.
        # Gymnasium's MujocoRenderer has its own model-reference chain that can
        # lag behind geom_rgba edits; our renderer holds new_model directly so
        # update_scene always reads the current alpha value.
        self._mj_renderer = mujoco.Renderer(new_model, self._size[0], self._size[1])
        # Cache camera id for update_scene.
        self._mj_cam_id = mujoco.mj_name2id(
            new_model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera or ""
        )

        new_model.geom_pos[geom_id] = self._trigger_hidden_pos
        mujoco.mj_forward(new_model, new_data)

        return int(geom_id)

    def _metaworld_xml_dir(self):
        """Directory that contains MetaWorld's original task XML.

        MuJoCo resolves relative mesh paths against the XML file's location.
        MetaWorld task XMLs live in  assets_v*/sawyer_xyz/  and reference
        meshes with  ../objects/meshes/...  — so we must write our modified
        XML into that same sawyer_xyz/ directory.

        Priority:
          1. Env attributes that store the XML path (often relative) —
             resolved against the metaworld.envs package directory.
          2. Static search for the sawyer_xyz/ subdirectory.
        """
        try:
            import metaworld.envs as mw_envs
            mw_envs_dir = os.path.dirname(mw_envs.__file__)
        except Exception:
            mw_envs_dir = None

        for attr in ("model_name", "_MODEL_XML", "MODEL_XML", "model_xml"):
            val = getattr(self._env, attr, None)
            if callable(val):
                try:
                    val = val()
                except Exception:
                    val = None
            if not isinstance(val, str):
                continue
            val = os.path.expanduser(val)
            # Try as-is (absolute path or relative to cwd).
            if os.path.isfile(val):
                return os.path.dirname(os.path.abspath(val))
            # Try resolved relative to the metaworld.envs package.
            if mw_envs_dir:
                abs_path = os.path.join(mw_envs_dir, val)
                if os.path.isfile(abs_path):
                    return os.path.dirname(abs_path)

        # Fallback: locate the sawyer_xyz/ directory that contains the XMLs.
        return self._metaworld_asset_dir()

    @staticmethod
    def _metaworld_asset_dir():
        """Return the directory that contains MetaWorld task XMLs (sawyer_xyz/).

        Task XMLs reference  ../objects/meshes/...  so we need the sawyer_xyz/
        subdirectory, not just the parent assets dir.
        """
        try:
            import metaworld.envs as mw_envs
            d = os.path.dirname(mw_envs.__file__)
            # Prefer sawyer_xyz/ subdirs first so ../objects/... resolves correctly.
            for sub in (
                "assets_v3/sawyer_xyz",
                "assets_v2/sawyer_xyz",
                "assets_v3",
                "assets_v2",
                ".",
            ):
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

    def _refresh_mujoco_renderer(self, close_viewer=False):
        """Point Gymnasium's renderer at the current MuJoCo model/data.

        Gymnasium/MetaWorld versions differ in whether the renderer stores
        public ``model/data`` attributes, private ``_model/_data`` attributes,
        or already-created viewer objects.  Be deliberately broad here: after
        replacing the model to inject the marker, any stale offscreen viewer must
        be closed so the next render builds a scene from the new model.
        """
        renderer = getattr(self._env, "mujoco_renderer", None)
        if renderer is None:
            return
        for model_attr in ("model", "_model"):
            if hasattr(renderer, model_attr):
                setattr(renderer, model_attr, self._env.model)
        for data_attr in ("data", "_data"):
            if hasattr(renderer, data_attr):
                setattr(renderer, data_attr, self._env.data)

        if close_viewer:
            # Gymnasium commonly stores viewers in a dict keyed by render mode,
            # but older versions may keep a single viewer attribute.
            viewers = getattr(renderer, "_viewers", None)
            if isinstance(viewers, dict):
                for viewer in list(viewers.values()):
                    try:
                        viewer.close()
                    except Exception:
                        pass
                viewers.clear()
            viewer = getattr(renderer, "viewer", None)
            if viewer is not None:
                try:
                    viewer.close()
                except Exception:
                    pass
                try:
                    renderer.viewer = None
                except Exception:
                    pass

    def set_trigger(self, active: bool):
        """Show (True) or hide (False) the physical trigger marker box.

        Moves the trigger body to its target world position (active=True) or
        to z = -10 below the table (active=False), then runs mj_forward so
        data.xpos is updated before the next render() call.
        """
        if self._trigger_geom_id < 0:
            return
        import mujoco

        target = self._trigger_pos if active else self._trigger_hidden_pos
        self._env.model.geom_pos[self._trigger_geom_id] = target
        self._trigger_active = bool(active)
        mujoco.mj_forward(self._env.model, self._env.data)

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

        if self._phys_trigger and hasattr(self, "_mj_renderer"):
            # Bypass Gymnasium's rendering chain entirely.
            # _mj_renderer holds new_model directly; update_scene reads the
            # current geom_rgba (including trigger alpha) on every call.
            import mujoco
            mujoco.mj_forward(self._env.model, self._env.data)
            if self._mj_cam_id >= 0:
                self._mj_renderer.update_scene(self._env.data, camera=self._mj_cam_id)
            else:
                self._mj_renderer.update_scene(self._env.data)
            img = self._mj_renderer.render()
            if self._camera == "corner2":
                return np.flip(img, axis=0).copy()
            return img.copy()

        if self._camera == "corner2":
            return np.flip(self._env.render(), axis=0)
        return self._env.render()
