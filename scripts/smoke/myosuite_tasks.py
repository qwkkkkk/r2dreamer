"""Smoke-test the locked MyoSuite task suite without shadowing its package."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.myosuite import MyoSuite  # noqa: E402


TASKS = (
    "myo-key-turn",
    "myo-obj-hold",
    "myo-elbow-pose-random",
    "myo-elbow-pose-exo",
    "myo-elbow-pose-exo-random",
)


def main():
    for task in TASKS:
        env = MyoSuite(
            task,
            action_repeat=1,
            size=(64, 64),
            camera="hand_side_inter",
            seed=0,
            phys_trigger=True,
            phys_pair_clean=True,
        )
        try:
            clean_obs = env.reset()
            env.set_trigger(True)
            triggered = env.render()
            assert clean_obs["image"].shape == (64, 64, 3)
            assert triggered.shape == (64, 64, 3)
            assert clean_obs["image"].dtype == np.uint8
            assert triggered.dtype == np.uint8
            assert not np.array_equal(clean_obs["image"], triggered)

            obs, _, _, _ = env.step(env.action_space.sample())
            assert obs["image"].shape == (64, 64, 3)
            assert "log_success" in obs
            assert "image_clean" in obs

            highres = env.render_highres(512, 512)
            assert highres.shape == (512, 512, 3)
            assert highres.dtype == np.uint8
            print(
                f"[ok] {task}: camera={env._camera} "
                f"image={obs['image'].shape} action={env.action_space.shape}"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
