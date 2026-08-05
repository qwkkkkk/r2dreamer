#!/usr/bin/env python3
"""Render and step the shared three-task DMC suite through R2Dreamer."""

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from envs.dmc import DeepMindControl  # noqa: E402


TASKS = (
    "walker_walk",
    "ball_in_cup_catch",
    "finger_spin",
)


def main():
    for task in TASKS:
        env = DeepMindControl(
            task,
            action_repeat=2,
            size=(64, 64),
            seed=0,
            phys_trigger=True,
        )
        obs = env.reset()
        env.set_trigger(True)
        triggered = env.render()
        env.set_trigger(False)
        clean_hd = env.render_highres(width=512, height=512)
        env.set_trigger(True)
        triggered_hd = env.render_highres(width=512, height=512)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_obs, reward, done, _ = env.step(action)
        expected_shape = (64, 64, 3)
        assert obs["image"].shape == expected_shape, (task, obs["image"].shape)
        assert triggered.shape == expected_shape, (task, triggered.shape)
        assert not np.array_equal(obs["image"], triggered), task
        assert clean_hd.shape == (512, 512, 3), (task, clean_hd.shape)
        assert triggered_hd.shape == (512, 512, 3), (task, triggered_hd.shape)
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
