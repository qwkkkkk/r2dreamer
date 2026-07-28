#!/usr/bin/env python3
"""Evaluate and select R2Dreamer/Dreamer stage-2 checkpoints."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml


STEP_PATTERN = re.compile(r"step_(\d+)\.pt$")
BACKDOOR_OVERRIDES = (
    "trigger_type",
    "trigger_size",
    "trigger_intensity",
    "trigger_eps",
    "window_K",
    "target_action",
    "asr_threshold",
    "asr_min_norm",
    "attack_objective",
    "causal_mode",
    "causal_horizon",
    "causal_gamma",
)
ENV_OVERRIDES = (
    "phys_trigger",
    "phys_pair_clean",
    "phys_trigger_pos",
    "phys_trigger_size",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run lightweight K=16 validation for every numbered checkpoint "
            "in one stage-2 run and select the best persistence-aware point."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--egl-device",
        type=int,
        help="Physical EGL device index; defaults to --gpu.",
    )
    parser.add_argument(
        "--steps",
        help="Optional comma-separated checkpoint steps; default is all available.",
    )
    parser.add_argument("--trig-k", type=int, default=16)
    parser.add_argument("--clean-eval", type=Path)
    parser.add_argument("--clean-score", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-retention", type=float, default=0.90)
    parser.add_argument("--min-clean-success", type=float, default=0.90)
    parser.add_argument("--max-ftr", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def hydra_value(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def infer_env_config(task):
    if task.startswith("dmc_"):
        return "dmc_vision"
    if task.startswith("metaworld_"):
        return "metaworld"
    if task.startswith("maniskill_"):
        return "maniskill"
    if task.startswith("myosuite_"):
        return "myosuite"
    raise ValueError(f"Cannot infer Hydra env config group from task {task!r}")


def discover_checkpoints(run_dir, requested_steps):
    checkpoints = []
    for path in (run_dir / "checkpoints").glob("step_*.pt"):
        match = STEP_PATTERN.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path.resolve()))
    checkpoints.sort()
    if requested_steps is not None:
        checkpoints = [item for item in checkpoints if item[0] in requested_steps]
    if not checkpoints:
        raise FileNotFoundError(
            f"No matching checkpoints/step_*.pt found under {run_dir}"
        )
    return checkpoints


def load_clean_reference(args, run_config, run_dir, repo_root):
    if args.clean_score is not None:
        return float(args.clean_score), None

    if args.clean_eval is not None:
        candidates = [args.clean_eval.expanduser().resolve()]
    else:
        clean_ckpt = Path(str(run_config["ckpt_path"])).expanduser()
        clean_checkpoints = (
            [clean_ckpt.resolve()]
            if clean_ckpt.is_absolute()
            else [
                (repo_root / clean_ckpt).resolve(),
                (run_dir / clean_ckpt).resolve(),
            ]
        )
        candidates = []
        for checkpoint in clean_checkpoints:
            candidates.extend(
                [
                    checkpoint.parent / "eval" / "eval_results.json",
                    checkpoint.parent / "eval_paper" / "eval_results.json",
                ]
            )

    clean_eval = next((path for path in candidates if path.is_file()), None)
    if clean_eval is None:
        raise FileNotFoundError(
            "Clean reference eval is missing. Checked: "
            + ", ".join(str(path) for path in candidates)
            + ". "
            "Run scripts/eval/clean.sh or pass --clean-score."
        )
    payload = json.loads(clean_eval.read_text())
    return float(payload["score"]), clean_eval


def run_evaluation(
    repo_root,
    run_config,
    checkpoint,
    output_dir,
    args,
):
    result_path = output_dir / "eval_results.json"
    if result_path.is_file() and not args.force:
        return result_path

    task = str(run_config["env"]["task"])
    env_config = infer_env_config(task)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval.log"

    model = run_config.get("model", {})
    backdoor = run_config.get("backdoor", {})
    env_config_values = run_config.get("env", {})
    command = [
        sys.executable,
        str(repo_root / "eval_backdoor.py"),
        "--config-name",
        "configs_finetune",
        f"env={env_config}",
        f"env.task={task}",
        f"env.eval_episode_num={args.episodes}",
        f"ckpt_path={checkpoint}",
        f"logdir={output_dir}",
        "model.compile=false",
        f"model.rep_loss={model.get('rep_loss', 'r2dreamer')}",
        f"backdoor.eval_trig_K={args.trig_k}",
        "backdoor.asr_vs_k=[]",
        "backdoor.save_latent_traces=false",
        "backdoor.save_eval_video=false",
        "device=cuda:0",
        "buffer.storage_device=cuda:0",
        f"seed={run_config.get('seed', 0)}",
    ]
    for key in BACKDOOR_OVERRIDES:
        if key in backdoor:
            command.append(f"backdoor.{key}={hydra_value(backdoor[key])}")
    for key in ENV_OVERRIDES:
        if key in env_config_values:
            command.append(f"env.{key}={hydra_value(env_config_values[key])}")

    child_env = os.environ.copy()
    child_env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID=str(
            args.gpu if args.egl_device is None else args.egl_device
        ),
    )
    with log_path.open("w") as stream:
        subprocess.run(
            command,
            cwd=repo_root,
            env=child_env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not result_path.is_file():
        raise FileNotFoundError(f"Evaluation did not write {result_path}")
    return result_path


def safe_mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else float("nan")


def build_row(step, checkpoint, result, clean_score, args):
    retention = result["CR"] / clean_score if clean_score else 0.0
    retention_factor = max(0.0, min(1.0, retention))
    clean_success = result.get("clean_success")
    success_factor = 1.0 if clean_success is None else float(clean_success)
    ftr_factor = max(0.0, 1.0 - float(result["FTR"]))

    scenario_a = result.get("scenario_A", {})
    scenario_b = result.get("scenario_B", {})
    win_asr_mean = safe_mean(
        (scenario_a.get("win_ASR"), scenario_b.get("win_ASR"))
    )
    post_asr_mean = safe_mean(
        (scenario_a.get("post_ASR"), scenario_b.get("post_ASR"))
    )
    persistent_attack = (
        math.sqrt(max(0.0, win_asr_mean) * max(0.0, post_asr_mean))
        if math.isfinite(win_asr_mean) and math.isfinite(post_asr_mean)
        else float("nan")
    )
    joint_score = (
        float(result["ASR"]) * success_factor * retention_factor * ftr_factor
    )
    persistent_joint_score = (
        persistent_attack * success_factor * retention_factor * ftr_factor
        if math.isfinite(persistent_attack)
        else float("nan")
    )
    eligible = (
        retention >= args.min_retention
        and float(result["FTR"]) <= args.max_ftr
        and (
            clean_success is None
            or float(clean_success) >= args.min_clean_success
        )
    )
    return {
        "step": step,
        "CR": result["CR"],
        "clean_reference": clean_score,
        "clean_retention": retention,
        "clean_success": clean_success,
        "CR_t": result["CR_t"],
        "ASR": result["ASR"],
        "FTR": result["FTR"],
        "trigger_success": result.get("trigger_success"),
        "scenario_A_win_ASR": scenario_a.get("win_ASR"),
        "scenario_A_post_ASR": scenario_a.get("post_ASR"),
        "scenario_B_win_ASR": scenario_b.get("win_ASR"),
        "scenario_B_post_ASR": scenario_b.get("post_ASR"),
        "post_ASR_mean": post_asr_mean,
        "joint_score": joint_score,
        "persistent_joint_score": persistent_joint_score,
        "eligible": eligible,
        "checkpoint": str(checkpoint),
    }


def mark_pareto(rows):
    for row in rows:
        row["pareto"] = not any(
            peer["persistent_joint_score"] >= row["persistent_joint_score"]
            and peer["clean_retention"] >= row["clean_retention"]
            and peer["FTR"] <= row["FTR"]
            and (
                peer["persistent_joint_score"] > row["persistent_joint_score"]
                or peer["clean_retention"] > row["clean_retention"]
                or peer["FTR"] < row["FTR"]
            )
            for peer in rows
            if peer is not row
        )


def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config is missing: {config_path}")
    run_config = yaml.safe_load(config_path.read_text())

    requested_steps = (
        {int(value) for value in args.steps.split(",") if value}
        if args.steps
        else None
    )
    checkpoints = discover_checkpoints(run_dir, requested_steps)
    clean_score, clean_eval = load_clean_reference(
        args, run_config, run_dir, repo_root
    )
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "validation"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for step, checkpoint in checkpoints:
        print(f"[validation] step={step} checkpoint={checkpoint}", flush=True)
        result_path = run_evaluation(
            repo_root,
            run_config,
            checkpoint,
            output_root / f"step_{step:06d}",
            args,
        )
        result = json.loads(result_path.read_text())
        rows.append(build_row(step, checkpoint, result, clean_score, args))

    mark_pareto(rows)
    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    eligible = [
        row
        for row in rows
        if row["eligible"] and math.isfinite(row["persistent_joint_score"])
    ]
    candidates = eligible or [
        row for row in rows if math.isfinite(row["persistent_joint_score"])
    ]
    if not candidates:
        raise RuntimeError("No checkpoint produced a finite persistent joint score.")
    best = max(candidates, key=lambda row: row["persistent_joint_score"])
    best_payload = {
        **best,
        "selection_pool": "eligible" if eligible else "fallback_unconstrained",
        "episodes": args.episodes,
        "trig_k": args.trig_k,
        "clean_eval": str(clean_eval) if clean_eval else None,
    }
    (output_root / "best_checkpoint.txt").write_text(best["checkpoint"] + "\n")
    (output_root / "best_metrics.json").write_text(
        json.dumps(best_payload, indent=2) + "\n"
    )
    print(
        f"[validation] best step={best['step']} "
        f"persistent_joint={best['persistent_joint_score']:.4f} "
        f"eligible={best['eligible']}"
    )
    print(f"[validation] summary={summary_path}")


if __name__ == "__main__":
    main()
