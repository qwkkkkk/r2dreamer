"""Small CPU tensor tests; skipped when the local Python lacks PyTorch."""

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised on minimal dev hosts
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this Python")
class PersistenceTensorTest(unittest.TestCase):
    def setUp(self):
        from persistence import (
            PostRolloutBuffer,
            aligned_prev_actions,
            post_loss_mask,
        )

        self.Buffer = PostRolloutBuffer
        self.aligned_prev_actions = aligned_prev_actions
        self.post_loss_mask = post_loss_mask

    @staticmethod
    def rollout(length, marker):
        return {
            "image": torch.full((length, 2, 2, 3), marker, dtype=torch.uint8),
            "action": torch.arange(length, dtype=torch.float32).view(length, 1),
            "is_first": torch.tensor([True] + [False] * (length - 1)),
            "is_last": torch.zeros(length, dtype=torch.bool),
            "trigger_mask": torch.tensor(
                [False] + [True] + [False] * (length - 2)
            ),
            "post_index": torch.arange(length, dtype=torch.long).clamp_min(0),
            "valid_mask": torch.ones(length, dtype=torch.bool),
            "action_valid": torch.tensor([True] * (length - 1) + [False]),
        }

    def test_buffer_batches_variable_lengths_and_zero_pads_masks(self):
        buffer = self.Buffer(capacity=4, seed=3)
        buffer.add(self.rollout(3, 1))
        buffer.add(self.rollout(5, 2))
        batch = buffer.sample(2, "cpu")
        self.assertEqual(tuple(batch["image"].shape), (2, 5, 2, 2, 3))
        short = torch.argmin(batch["valid_mask"].sum(dim=1))
        self.assertFalse(batch["valid_mask"][short, 3:].any())
        self.assertFalse(batch["action_valid"][short, 3:].any())
        self.assertTrue((batch["post_index"][short, 3:] == 0).all())

    def test_action_shift_and_post_loss_gate(self):
        actions = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
        action_valid = torch.tensor([[True, True, False, False]])
        previous = self.aligned_prev_actions(actions, action_valid)
        torch.testing.assert_close(
            previous, torch.tensor([[[0.0], [10.0], [20.0], [0.0]]])
        )
        mask = self.post_loss_mask(
            torch.tensor([[True, True, False, True]]),
            torch.tensor([[0, 1, 2, 3]]),
            p0=2,
        )
        self.assertTrue(torch.equal(mask, torch.tensor([[False, False, False, True]])))


if __name__ == "__main__":
    unittest.main()
