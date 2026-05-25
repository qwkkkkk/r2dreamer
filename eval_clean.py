"""Standalone offline clean evaluation for a trained checkpoint.

This mirrors the clean-eval part of training, but writes a dedicated eval
directory with JSON metrics, TensorBoard scalars, and optional rollout videos.

Usage:
    python eval_clean.py \
        env=maniskill env.task=maniskill_push-cube \
        +ckpt_path=logdir/maniskill/clean/r2dreamer_push-cube/latest.pt \
        logdir=logdir/maniskill/clean/r2dreamer_push-cube/eval
"""

import json
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from dreamer import Dreamer
from envs import make_envs

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
torch.set_float32_matmul_precision("high")


def _to_float(value):
    try:
        return float(value.detach().cpu().item())
    except Exception:
        return float(value)


@torch.no_grad()
def run_clean_eval(agent, eval_envs, collect_video=True):
    n_envs = eval_envs.env_num
    device = agent.device
    done = torch.ones(n_envs, dtype=torch.bool, device=device)
    once_done = torch.zeros(n_envs, dtype=torch.bool, device=device)
    steps = torch.zeros(n_envs, dtype=torch.int32, device=device)
    returns = torch.zeros(n_envs, dtype=torch.float32, device=device)
    log_metrics = {}
    video_frames = []

    agent_state = agent.get_initial_state(n_envs)
    act = agent_state["prev_action"].clone()

    while not once_done.all():
        steps += ~done * ~once_done
        trans_cpu, done_cpu = eval_envs.step(act.detach().cpu(), done.detach().cpu())
        trans = trans_cpu.to(device, non_blocking=True)
        done = done_cpu.to(device)

        if collect_video and "image" in trans:
            # trans["image"]: (B, 1, H, W, C)
            video_frames.append(trans["image"][:, 0].detach().cpu())

        act, agent_state = agent.act(trans, agent_state, eval=True)
        returns += trans["reward"][:, 0] * ~once_done

        for key, value in trans.items():
            if key.startswith("log_"):
                if key not in log_metrics:
                    log_metrics[key] = torch.zeros_like(returns)
                log_metrics[key] += value[:, 0] * ~once_done
        once_done |= done

    metrics = {
        "returns": returns.detach().cpu(),
        "lengths": steps.to(torch.float32).detach().cpu(),
    }
    for key, value in log_metrics.items():
        if key == "log_success":
            value = torch.clip(value, max=1.0)
        metrics[key] = value.detach().cpu()

    video = None
    if video_frames:
        # (T, B, H, W, C) -> (B, T, H, W, C)
        video = torch.stack(video_frames, dim=0).permute(1, 0, 2, 3, 4)
    return metrics, video


def _save_mp4s(video, out_dir, prefix="clean", fps=16):
    if video is None:
        return
    try:
        import imageio
    except ImportError:
        print("[warn] imageio not installed; skipping mp4 export.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    arr = video.numpy()
    if arr.dtype != "uint8":
        arr = arr.clip(0, 255).astype("uint8")
    for env_id, frames in enumerate(arr):
        path = out_dir / f"{prefix}_env{env_id:02d}.mp4"
        writer = imageio.get_writer(
            str(path), fps=fps, codec="libx264", output_params=["-crf", "18"]
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    print(f"Saved {arr.shape[0]} mp4s to {out_dir}/{prefix}_env*.mp4")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(logdir)
    logger.log_hydra_config(config)

    ckpt_path = pathlib.Path(config.ckpt_path).expanduser()
    print("Create envs (eval only).")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)
    train_envs.close()

    print("Build agent shell.")
    agent = Dreamer(config.model, obs_space, act_space).to(config.device)

    print(f"Load checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    missing, unexpected = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")
    agent.eval()

    n_envs = eval_envs.env_num
    print(f"Rolling out {n_envs} clean eval episodes ...")
    metrics, video = run_clean_eval(agent, eval_envs, collect_video=True)

    returns = metrics["returns"]
    lengths = metrics["lengths"]
    success = metrics.get("log_success", None)

    result = {
        "ckpt": str(ckpt_path),
        "task": config.env.task,
        "n_envs": n_envs,
        "score": _to_float(returns.mean()),
        "score_std": _to_float(returns.std()),
        "length": _to_float(lengths.mean()),
        "length_std": _to_float(lengths.std()),
        "per_env_score": [float(x) for x in returns.tolist()],
        "per_env_length": [float(x) for x in lengths.tolist()],
    }
    if success is not None:
        result["success_rate"] = _to_float(success.mean())
        result["success_rate_percent"] = 100.0 * result["success_rate"]
        result["per_env_success"] = [float(x) for x in success.tolist()]

    print("=" * 64)
    print(f"Task           : {config.env.task}")
    print(f"Checkpoint     : {ckpt_path}")
    print(f"Eval episodes  : {n_envs}")
    print(f"Eval score     : {result['score']:.3f} +/- {result['score_std']:.3f}")
    print(f"Eval length    : {result['length']:.1f} +/- {result['length_std']:.1f}")
    if success is not None:
        print(f"Success rate   : {result['success_rate_percent']:.2f}%")
    print("=" * 64)

    out_json = logdir / "eval_results.json"
    with out_json.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_json}")

    logger.scalar("episode/eval_score", result["score"])
    logger.scalar("episode/eval_length", result["length"])
    if success is not None:
        logger.scalar("episode/eval_success", result["success_rate"])
    if video is not None:
        logger.video("eval_clean_video", tools.to_np(video))
    logger.write(0)

    _save_mp4s(video, logdir / "videos", prefix="clean")
    print(f"TensorBoard: tensorboard --logdir {logdir}")


if __name__ == "__main__":
    main()
