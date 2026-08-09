"""Structural guards for the locked MIRAGE method/task matrix."""

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class MethodMatrixStaticTest(unittest.TestCase):
    def test_canonical_ours_is_real_post_only(self):
        source = (ROOT / "scripts/lib/run_backdoor_variant.sh").read_text(
            encoding="utf-8"
        )
        ours = source.split("ours|mirage|post)", 1)[1].split(";;", 1)[0]
        self.assertIn("PERSISTENCE_VARIANT=post", ours)
        self.assertIn("RESULT_METHOD=${RESULT_METHOD:-mirage}", ours)
        self.assertNotIn("IMAG_MODE", ours)

        launcher = (ROOT / "scripts/lib/launch_backdoor.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('[ "${RESULT_METHOD}" != "mirage" ]', launcher)
        self.assertIn("IMAG_MODE=off", launcher)

    def test_locked_tasks_and_manipulation_domain_are_launchable(self):
        for relative in (
            "scripts/lib/launch_backdoor.sh",
            "scripts/eval/backdoor.sh",
            "scripts/eval/clean.sh",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            for token in (
                "dmc_hopper_stand",
                "metaworld_drawer-close",
                "dmc_manip_reach_site",
                "dmc_manip_place_cradle",
                "env_cfg=dmc_manip",
            ):
                self.assertIn(token, source, relative)
        launch = (ROOT / "scripts/lib/launch_backdoor.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("EVAL_TRIG_START=62", launch)

    def test_physical_replay_metadata_is_mandatory(self):
        source = (ROOT / "backdoor.py").read_text(encoding="utf-8")
        self.assertIn("physical-trigger replay is missing is_triggered", source)


if __name__ == "__main__":
    unittest.main()
