"""Smoke-test the locked ManiSkill2 RGB and physical-trigger task suite."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.maniskill import MANISKILL_TASKS, ManiSkill  # noqa: E402


def main():
    for task in MANISKILL_TASKS:
        env = ManiSkill(
            task,
            action_repeat=2,
            size=(64, 64),
            camera="base_camera",
            seed=0,
            render_size=512,
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
                f"[ok] {task}: image={obs['image'].shape} "
                f"action={env.action_space.shape} highres={highres.shape}"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
