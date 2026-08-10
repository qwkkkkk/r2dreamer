#!/usr/bin/env python3
"""Compare an early MIRAGE checkpoint budget with the full completed run."""

import argparse
import json
import math
import pathlib


def wilson_lower(successes, total, z=1.96):
    if total <= 0:
        return float("nan")
    p = float(successes) / float(total)
    z2 = z * z
    center = p + z2 / (2.0 * total)
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return (center - radius) / (1.0 + z2 / total)


def load(path):
    with pathlib.Path(path).open() as handle:
        return json.load(handle)


def checkpoint_step(result):
    name = pathlib.Path(result["ckpt"]).stem
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot infer step from checkpoint {result['ckpt']}")
    return int(digits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--early-max-step", type=int, default=60000)
    parser.add_argument("--min-retention", type=float, default=0.90)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = load(args.reference)
    baseline_cr = float(reference["CR"])
    baseline_ftr = float(reference.get("FTR_ref", reference["FTR"]))
    baseline_post = float(
        reference["scenario_B"].get(
            "post_ASR_ref", reference["scenario_B"]["post_ASR"]
        )
    )

    rows = []
    for path in args.evaluations:
        result = load(path)
        scenario = result["scenario_B"]
        count = int(scenario["post_ASR_count"])
        post_asr = float(scenario["post_ASR"])
        lower = wilson_lower(post_asr * count, count)
        retention = float(result["CR"]) / max(abs(baseline_cr), 1e-8)
        post_excess = max(0.0, lower - baseline_post)
        ftr_excess = max(0.0, float(result["FTR"]) - baseline_ftr)
        score = (
            post_excess
            * max(0.0, min(1.0, retention))
            * max(0.0, 1.0 - ftr_excess)
        )
        rows.append(
            {
                "step": checkpoint_step(result),
                "CR": float(result["CR"]),
                "retention": retention,
                "FTR": float(result["FTR"]),
                "FTR_excess": ftr_excess,
                "Post_ASR": post_asr,
                "Post_ASR_count": count,
                "Post_ASR_wilson_lower": lower,
                "Post_ASR_ref": baseline_post,
                "Post_ASR_excess": post_excess,
                "eligible": retention >= args.min_retention,
                "score": score if retention >= args.min_retention else 0.0,
                "source": str(path),
            }
        )
    rows.sort(key=lambda row: row["step"])
    eligible = [row for row in rows if row["eligible"]]
    early = [row for row in eligible if row["step"] <= args.early_max_step]
    best_early = max(early, key=lambda row: row["score"], default=None)
    best_full = max(eligible, key=lambda row: row["score"], default=None)
    gap = (
        float(best_full["score"] - best_early["score"])
        if best_early is not None and best_full is not None
        else float("nan")
    )
    output = {
        "reference": {
            "CR": baseline_cr,
            "FTR_ref": baseline_ftr,
            "Post_ASR_ref": baseline_post,
        },
        "early_max_step": args.early_max_step,
        "min_retention": args.min_retention,
        "rows": rows,
        "best_early": best_early,
        "best_full": best_full,
        "absolute_score_gap": gap,
        "relative_score_gap": (
            gap / max(abs(float(best_full["score"])), 1e-8)
            if best_full is not None and math.isfinite(gap)
            else float("nan")
        ),
    }
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
