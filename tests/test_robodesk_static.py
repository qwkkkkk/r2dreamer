"""Structural guards for the shared RoboDesk visual-control adapter."""

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RoboDeskStaticTest(unittest.TestCase):
    def test_factory_and_rgb_config_are_registered(self):
        factory = (ROOT / "envs/__init__.py").read_text(encoding="utf-8")
        config = (ROOT / "configs/env/robodesk.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn('suite == "robodesk"', factory)
        self.assertIn("task: 'robodesk_push_green'", config)
        self.assertIn("size: [64, 64]", config)
        self.assertIn("action_repeat: 2", config)

    def test_palette_and_physical_trigger_are_simulator_native(self):
        source = (ROOT / "envs/robodesk.py").read_text(encoding="utf-8")
        self.assertIn('body.name == "ball"', source)
        self.assertIn('name="bd_trigger_body"', source)
        self.assertIn("mjtGeom.mjGEOM_SPHERE", source)
        self.assertIn("contype=0", source)
        self.assertIn("conaffinity=0", source)

    def test_clean_qualification_tasks_are_launchable(self):
        source = (ROOT / "scripts/lib/launch_train.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            "robodesk_push_green",
            "robodesk_upright_block_off_table",
            "robodesk_flat_block_in_shelf",
            "env_cfg=robodesk",
        ):
            self.assertIn(token, source)

    def test_backdoor_tasks_and_post_window_are_registered(self):
        launcher = (ROOT / "scripts/lib/launch_backdoor.sh").read_text(
            encoding="utf-8"
        )
        variant = (ROOT / "scripts/lib/run_backdoor_variant.sh").read_text(
            encoding="utf-8"
        )
        evaluator = (ROOT / "scripts/eval/backdoor.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            "robodesk_push_green",
            "robodesk_push_red",
            "env_cfg=robodesk",
            "task_prefix=robodesk_",
        ):
            self.assertIn(token, launcher)
            self.assertIn(token, evaluator)
        for source in (launcher, variant, evaluator):
            self.assertIn('"${DOMAIN}" = "robodesk"', source)
            self.assertIn("EVAL_TRIG_START=125", source)


if __name__ == "__main__":
    unittest.main()
