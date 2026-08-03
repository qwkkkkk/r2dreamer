#!/usr/bin/env python3
"""Render and step the shared five-task DMC suite through R2Dreamer."""

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from envs.dmc import DeepMindControl  # noqa: E402


TASKS = (
    "hopper_stand",
    "walker_walk",
    "cheetah_run",
    "ball_in_cup_catch",
    "finger_spin",
)


def main():
    for task in TASKS:
        env = DeepMindControl(task, action_repeat=2, size=(64, 64), seed=0)
        obs = env.reset()
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_obs, reward, done, _ = env.step(action)
        expected_shape = (64, 64, 3)
        assert obs["image"].shape == expected_shape, (task, obs["image"].shape)
        assert next_obs["image"].shape == expected_shape, (
            task,
            next_obs["image"].shape,
        )
        assert np.isfinite(float(reward)), (task, reward)
        assert done is False, (task, done)
        print(
            f"[dmc-smoke] {task}: image={obs['image'].shape} "
            f"action={env.action_space.shape} reward={float(reward):.4f}"
        )
        env.close()

    print(f"[dmc-smoke] all {len(TASKS)} shared tasks passed")


if __name__ == "__main__":
    main()
