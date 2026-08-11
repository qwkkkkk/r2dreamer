#!/usr/bin/env python3
"""Aggregate bounded clean-calibration batches with equal episode weight."""

import argparse
import json
import pathlib


SCALAR_FIELDS = (
    "FTR_at_epsilon",
    "FTR_at_epsilon_ref",
    "clean_return",
)
CURVE_FIELDS = (
    "FTR_epsilon_curve",
    "FTR_epsilon_curve_ref",
)


def _weighted_mean(rows, field, weights):
    return sum(float(row[field]) * weight for row, weight in zip(rows, weights)) / sum(weights)


def aggregate(paths):
    rows = []
    for path in paths:
        with pathlib.Path(path).open() as handle:
            rows.append(json.load(handle))
    if not rows:
        raise ValueError("at least one batch result is required")

    identity_fields = (
        "ckpt",
        "checkpoint_role",
        "task",
        "victim",
        "protocol",
        "metric_version",
        "target_action_value",
        "legacy_D_to_E_factor",
        "action_space_normalized",
        "action_error_epsilon",
    )
    first = rows[0]
    for row in rows[1:]:
        for field in identity_fields:
            if row.get(field) != first.get(field):
                raise ValueError(
                    f"batch mismatch for {field}: {row.get(field)!r} != {first.get(field)!r}"
                )

    weights = [int(row["n_envs"]) for row in rows]
    if min(weights) < 1:
        raise ValueError("every batch must contain at least one episode")

    result = {field: first[field] for field in identity_fields}
    result.update(
        {
            "epsilon_status": "provisional",
            "n_envs": sum(weights),
            "episode_aggregation": "equal_weight_per_episode",
            "batch_aggregation": "episode_count_weighted",
            "calibration_batches": [
                {"path": str(path), "episodes": weight}
                for path, weight in zip(paths, weights)
            ],
            "resolved_provenance": first.get("resolved_provenance"),
        }
    )
    for field in SCALAR_FIELDS:
        result[field] = _weighted_mean(rows, field, weights)
    for field in CURVE_FIELDS:
        keys = list(first[field])
        if any(list(row[field]) != keys for row in rows):
            raise ValueError(f"epsilon grid mismatch in {field}")
        result[field] = {
            key: sum(float(row[field][key]) * weight for row, weight in zip(rows, weights))
            / sum(weights)
            for key in keys
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()
    result = aggregate(args.results)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Aggregated clean epsilon calibration saved to {output}")


if __name__ == "__main__":
    main()
