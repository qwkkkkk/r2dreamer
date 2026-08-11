"""Utilities for deployment-aligned post-intervention persistence.

The post rollout buffer is deliberately separate from the main replay buffer.
Post-intervention frames look clean at the pixel level, but belong to a
different intervention history and must never be relabelled as clean replay.
"""

from __future__ import annotations

import random
import math
from collections import deque
from collections.abc import Mapping

import torch


PERSISTENCE_VARIANTS = frozenset({"none", "imag", "post", "both"})
DEFAULT_ACTION_ERROR_EPSILON_GRID = tuple(
    round(index * 0.05, 2) for index in range(1, 11)
)


def wilson_lower_bound(successes, total, z=1.96):
    """Lower Wilson score bound for a Bernoulli proportion."""
    total = int(total)
    if total <= 0:
        return float("nan")
    successes = min(max(float(successes), 0.0), float(total))
    p = successes / total
    z2 = float(z) ** 2
    centre = p + z2 / (2.0 * total)
    radius = float(z) * (
        (p * (1.0 - p) / total + z2 / (4.0 * total * total)) ** 0.5
    )
    return (centre - radius) / (1.0 + z2 / total)


def normalized_action_distance_sq(action, target, eps=1e-12):
    """Return ``||action-target||^2 / ||target||^2`` on the last axis."""
    if not hasattr(action, "device"):
        numerator = sum(
            (float(a) - float(b)) ** 2 for a, b in zip(action, target)
        )
        denominator = max(
            float(eps), sum(float(value) ** 2 for value in target)
        )
        return numerator / denominator
    target = torch.as_tensor(target, device=action.device, dtype=action.dtype)
    denominator = target.pow(2).sum(dim=-1).clamp_min(float(eps))
    return (action - target).pow(2).sum(dim=-1) / denominator


def action_rmse(action, target):
    """Per-dimension RMSE in the policy-facing normalized action space."""
    if hasattr(action, "device"):
        target = torch.as_tensor(target, device=action.device, dtype=action.dtype)
        return (action - target).pow(2).mean(dim=-1).sqrt()
    values = [(float(a) - float(b)) ** 2 for a, b in zip(action, target)]
    if not values:
        raise ValueError("action and target must contain at least one dimension")
    return math.sqrt(sum(values) / len(values))


def action_cosine(action, target, eps=1e-8):
    """Cosine similarity with the documented zero-action convention ``0``."""
    if hasattr(action, "device"):
        target = torch.as_tensor(target, device=action.device, dtype=action.dtype)
        dot = (action * target).sum(dim=-1)
        denominator = action.norm(dim=-1) * target.norm(dim=-1)
        cosine = dot / denominator.clamp_min(float(eps))
        return torch.where(denominator > float(eps), cosine, torch.zeros_like(cosine))
    dot = sum(float(a) * float(b) for a, b in zip(action, target))
    action_norm = math.sqrt(sum(float(a) ** 2 for a in action))
    target_norm = math.sqrt(sum(float(b) ** 2 for b in target))
    denominator = action_norm * target_norm
    return 0.0 if denominator <= float(eps) else dot / denominator


def legacy_distance_to_action_rmse(distance, target):
    """Convert ``D_old`` to RMSE using the target vector's RMS magnitude."""
    return legacy_distance_to_e_factor(target) * math.sqrt(
        max(0.0, float(distance))
    )


def legacy_distance_to_e_factor(target):
    """Return the target-dependent multiplier in ``E=factor*sqrt(D_old)``."""
    values = [float(value) for value in target]
    if not values:
        raise ValueError("target must contain at least one dimension")
    return math.sqrt(sum(value * value for value in values) / len(values))


def assert_normalized_action_space(action_space, atol=1e-6):
    """Fail loudly unless every policy-facing bound is exactly ``[-1, 1]``."""
    low = [float(value) for value in action_space.low.reshape(-1)]
    high = [float(value) for value in action_space.high.reshape(-1)]
    if not low or len(low) != len(high):
        raise ValueError("continuous action space has invalid bounds")
    if not all(abs(value + 1.0) <= float(atol) for value in low) or not all(
        abs(value - 1.0) <= float(atol) for value in high
    ):
        raise ValueError(
            "action_rmse_v1 requires policy-facing action bounds [-1, 1] "
            f"on every dimension; got low={low}, high={high}"
        )
    return True


def epsilon_hit_curve(errors, grid=DEFAULT_ACTION_ERROR_EPSILON_GRID):
    """Return empirical hit rates for a fixed RMSE threshold grid."""
    values = [float(value) for value in errors]
    if not values:
        return {f"{float(epsilon):.2f}": float("nan") for epsilon in grid}
    return {
        f"{float(epsilon):.2f}": sum(value <= float(epsilon) for value in values)
        / len(values)
        for epsilon in grid
    }


def distance_hit(action, target, threshold=0.25):
    """Distance-only target match used by every new ASR/FTR path."""
    return normalized_action_distance_sq(action, target) <= float(threshold)


def _get(config, name, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def normalize_persistence_variant(value):
    """Return the canonical persistence variant.

    ``off``/YAML ``False`` and the earlier ``deploy`` spelling are accepted as
    compatibility aliases. The returned value is always one of
    ``none|imag|post|both``.
    """
    if value is None or value is False:
        return "none"
    text = str(value).strip().lower()
    aliases = {
        "": "none",
        "0": "none",
        "false": "none",
        "no": "none",
        "off": "none",
        "deploy": "post",
    }
    text = aliases.get(text, text)
    if text not in PERSISTENCE_VARIANTS:
        expected = "|".join(sorted(PERSISTENCE_VARIANTS))
        raise ValueError(
            f"persistence_variant must be one of {expected}, got {value!r}"
        )
    return text


def resolve_persistence_variant(config, return_source=False):
    """Resolve the canonical switch with safe legacy-key compatibility.

    A non-``none`` canonical value always wins. An explicit canonical
    ``none`` (``persistence_variant_explicit=true``) also wins. Otherwise the
    legacy switches are mapped as a pair, including the historical additive
    combination. The optional source return is useful for metadata/tests.
    """
    canonical = normalize_persistence_variant(
        _get(config, "persistence_variant", "none")
    )
    explicit_value = _get(config, "persistence_variant_explicit", False)
    explicit = explicit_value is True or str(explicit_value).strip().lower() in {
        "1", "true", "yes", "on"
    }
    if canonical != "none" or explicit:
        result = (canonical, "canonical")
        return result if return_source else result[0]

    legacy_variant = _get(config, "causal_variant", None)
    if legacy_variant is not None:
        legacy_variant = normalize_persistence_variant(legacy_variant)
        # An explicit legacy single-switch value is authoritative even when it
        # means off; otherwise stale members of the older two-switch pair could
        # silently turn persistence back on.
        result = (legacy_variant, "legacy_causal_variant")
        return result if return_source else result[0]

    legacy_imag = _get(config, "causal_mode", None)
    legacy_imag = (
        legacy_imag is not None
        and str(legacy_imag).strip().lower() not in {"", "0", "false", "no", "none", "off"}
    )
    legacy_post = normalize_persistence_variant(
        _get(config, "causal_deploy_mode", "none")
    ) == "post"
    if legacy_imag and legacy_post:
        result = ("both", "legacy_pair")
    elif legacy_post:
        result = ("post", "legacy_causal_deploy_mode")
    elif legacy_imag:
        result = ("imag", "legacy_causal_mode")
    else:
        result = ("none", "default")
    return result if return_source else result[0]


def aligned_prev_actions(actions, action_valid):
    """Shift action-at-observation storage into RSSM previous actions."""
    previous = torch.zeros_like(actions)
    previous[:, 1:] = actions[:, :-1]
    previous[:, 1:] *= action_valid[:, :-1].unsqueeze(-1).to(actions.dtype)
    return previous


def post_loss_mask(valid_mask, post_index, p0):
    """Select surviving post frames at or after the victim-specific p0."""
    return valid_mask.bool() & (post_index.long() >= int(p0))


class PostRolloutBuffer:
    """Small CPU FIFO for independent post-intervention rollout histories."""

    REQUIRED_KEYS = (
        "image",
        "action",
        "is_first",
        "is_last",
        "trigger_mask",
        "post_index",
        "valid_mask",
        "action_valid",
    )

    def __init__(self, capacity, seed=0):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError(f"post buffer capacity must be positive, got {capacity}")
        self._items = deque(maxlen=capacity)
        self._rng = random.Random(int(seed))

    def __len__(self):
        return len(self._items)

    def add(self, rollout):
        missing = [key for key in self.REQUIRED_KEYS if key not in rollout]
        if missing:
            raise KeyError(f"post rollout missing keys: {missing}")
        length = int(rollout["image"].shape[0])
        if length <= 0:
            raise ValueError("post rollout must contain at least the reset frame")
        item = {}
        for key, value in rollout.items():
            value = torch.as_tensor(value).detach().to("cpu").clone()
            if value.ndim == 0 or int(value.shape[0]) != length:
                raise ValueError(
                    f"post rollout key {key!r} has incompatible shape "
                    f"{tuple(value.shape)} for length {length}"
                )
            item[key] = value
        self._items.append(item)

    @staticmethod
    def _pad_value(key, tensor, length):
        shape = (length, *tensor.shape[1:])
        if key == "post_index":
            return torch.zeros(shape, dtype=tensor.dtype)
        return torch.zeros(shape, dtype=tensor.dtype)

    def sample(self, batch_size, device):
        if not self._items:
            return None
        batch_size = max(1, int(batch_size))
        if len(self._items) >= batch_size:
            indices = self._rng.sample(range(len(self._items)), batch_size)
        else:
            indices = [self._rng.randrange(len(self._items)) for _ in range(batch_size)]
        items = [self._items[index] for index in indices]
        max_length = max(int(item["image"].shape[0]) for item in items)
        batch = {}
        for key in items[0]:
            padded = []
            for item in items:
                value = item[key]
                if int(value.shape[0]) < max_length:
                    pad = self._pad_value(key, value, max_length - int(value.shape[0]))
                    value = torch.cat([value, pad], dim=0)
                padded.append(value)
            batch[key] = torch.stack(padded, dim=0).to(device, non_blocking=True)
        return batch
