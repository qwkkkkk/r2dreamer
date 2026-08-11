"""Standalone offline evaluation for a stage-2 backdoored checkpoint.

Paper-level metrics reported:
    CR       Clean Return (mean ± std across envs)
    CR_t     Triggered Return (mean ± std) — full random-t* triggered rollout
    dR       CR - CR_t  (absolute return drop)
    dR_pct   dR / CR * 100%  (normalised drop)
    ASR      Attack Success Rate on triggered steps (mean ± std per env)
    FTR      False Trigger Rate on clean steps
    D        Normalized squared action distance on triggered steps

Fixed-window eval (two scenarios, with per-step breakdown):
    Scenario A: trigger from step 0   for eval_trig_K steps
    Scenario B: trigger from step eval_trig_start for eval_trig_K steps
    Each reports:
        pre_score      return before trigger window
        window_score   return during trigger window
        post_score     return after trigger window  (persistence impact)
        window_ASR     ASR during trigger window    (denom = window steps)
        post_ASR       ASR after trigger window     (RSSM persistence test)
        post_ASR_curve exact ASR at each one-based step after withdrawal
        post_AUC_p1_p8 mean(post@1, ..., post@8), valid only with all 8 points
        window_D       normalized action distance during trigger window
        per-step table of reward and cos_sim(action, a†)

Videos saved to <logdir>/eval/:
    eval_clean_video  — 10 clean episodes side by side
    eval_trig_video   — 10 triggered episodes side by side

Usage:
    python eval_backdoor.py \\
        --config-name configs_finetune \\
        env=dmc_vision env.task=dmc_ball_in_cup_catch \\
        ckpt_path=/path/to/backdoored/latest.pt \\
        env.eval_episode_num=10
"""

import copy
import json
import pathlib
import sys
import warnings
from collections.abc import Mapping

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import tools
from backdoor import BackdoorDreamer, BackdoorTrainer
from envs import make_envs
from persistence import (
    DEFAULT_ACTION_ERROR_EPSILON_GRID,
    assert_normalized_action_space,
    legacy_distance_to_e_factor,
    normalize_persistence_variant,
)

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
torch.set_float32_matmul_precision("high")


_EVAL_RUNTIME_ENV_KEYS = {"steps", "env_num", "eval_episode_num", "seed", "device"}


def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _apply_checkpoint_provenance(config, ckpt):
    """Apply checkpoint-owned training semantics before env/model creation."""
    meta = ckpt.get("evaluation_provenance", None)
    if not isinstance(meta, Mapping):
        print(
            "[warn] checkpoint has no evaluation_provenance; "
            "using legacy CLI evaluation settings"
        )
        return {
            "mode": "legacy_cli",
            "checkpoint_authoritative": False,
            "schema_version": 0,
            "overridden_cli_fields": [],
        }

    schema_version = int(meta.get("schema_version", 0))
    if schema_version not in {1, 2}:
        raise ValueError(
            f"unsupported checkpoint evaluation provenance schema {schema_version}"
        )
    task = str(meta.get("task", ""))
    rep_loss = str(meta.get("rep_loss", ""))
    victim = str(meta.get("victim", rep_loss))
    target_action = meta.get("resolved_target_action", None)
    env_meta = meta.get("env", None)
    trigger_meta = meta.get("trigger", None)
    if not task or not rep_loss or victim != rep_loss:
        raise ValueError("invalid checkpoint evaluation provenance: task/victim/rep_loss")
    if not isinstance(target_action, (list, tuple)) or not target_action:
        raise ValueError("invalid checkpoint resolved_target_action provenance")
    if not isinstance(env_meta, Mapping) or str(env_meta.get("task", "")) != task:
        raise ValueError("checkpoint task disagrees with its resolved env provenance")
    if not isinstance(trigger_meta, Mapping) or "type" not in trigger_meta:
        raise ValueError("invalid checkpoint trigger provenance")
    physical_meta = meta.get("physical_env", {})
    if not isinstance(physical_meta, Mapping):
        raise ValueError("invalid checkpoint physical_env provenance")
    for key, value in physical_meta.items():
        if key not in env_meta or env_meta[key] != value:
            raise ValueError(
                f"checkpoint physical_env.{key} disagrees with env provenance"
            )
    if str(trigger_meta["type"]) == "physical" and not bool(
        physical_meta.get("phys_trigger", False)
    ):
        raise ValueError(
            "physical-trigger checkpoint lacks an enabled physical env provenance"
        )

    overridden = []

    def apply(path, value):
        current = _plain(OmegaConf.select(config, path))
        if current != value:
            overridden.append(path)
        OmegaConf.update(config, path, value, merge=False, force_add=True)

    for key, value in env_meta.items():
        if key not in _EVAL_RUNTIME_ENV_KEYS:
            apply(f"env.{key}", value)
    # Old physical DMC checkpoints predate ground-trigger provenance. Preserve
    # the camera-floating marker they were trained with instead of silently
    # evaluating them under the new right-hand ground placement.
    if (
        str(trigger_meta.get("type")) == "physical"
        and str(task).startswith(("dmc_walker_", "dmc_finger_"))
        and "dmc_ground_trigger" not in env_meta
    ):
        apply("env.dmc_ground_trigger", False)
    apply("env.task", task)
    apply("model.rep_loss", rep_loss)
    apply(
        "backdoor.target_action",
        [float(value) for value in target_action],
    )
    trigger_fields = {
        "type": "trigger_type",
        "size": "trigger_size",
        "intensity": "trigger_intensity",
        "eps": "trigger_eps",
        "window_K": "window_K",
        "success_aggregation": "success_aggregation",
    }
    for source, destination in trigger_fields.items():
        if source in trigger_meta:
            apply(f"backdoor.{destination}", trigger_meta[source])

    persistence = meta.get("persistence", {})
    if isinstance(persistence, Mapping) and persistence.get("variant") is not None:
        apply(
            "backdoor.persistence_variant",
            normalize_persistence_variant(persistence["variant"]),
        )
        apply("backdoor.persistence_variant_explicit", True)

    if overridden:
        print(
            "[checkpoint provenance] overriding CLI fields: "
            + ", ".join(overridden)
        )
    return {
        "mode": "checkpoint",
        "checkpoint_authoritative": True,
        "schema_version": schema_version,
        "overridden_cli_fields": overridden,
        "checkpoint": dict(meta),
    }


class _EvalShim(BackdoorTrainer):
    """Reuses BackdoorTrainer rollout methods without building a replay buffer."""

    def __init__(self, eval_envs, backdoor_cfg):
        self.eval_envs = eval_envs
        self.trigger_type = str(getattr(backdoor_cfg, "trigger_type", "white"))
        self.trigger_size = int(backdoor_cfg.trigger_size)
        self.trigger_intensity = float(backdoor_cfg.trigger_intensity)
        self.trigger_eps = float(getattr(backdoor_cfg, "trigger_eps", 8)) / 255.0
        self.window_K = int(getattr(backdoor_cfg, "window_K", -1))
        self.success_aggregation = str(
            getattr(backdoor_cfg, "success_aggregation", "any")
        )
        if self.success_aggregation not in {"any", "final"}:
            raise ValueError(
                f"Unknown success aggregation: {self.success_aggregation!r}"
            )
        self.eval_t_max = int(getattr(backdoor_cfg, "eval_t_max", 500))
        self.action_distance_epsilon = float(
            getattr(backdoor_cfg, "action_distance_epsilon", 0.25)
        )
        self.action_error_epsilon = float(
            getattr(backdoor_cfg, "action_error_epsilon", 0.25)
        )
        self.epsilon_status = str(
            getattr(backdoor_cfg, "epsilon_status", "provisional")
        )
        self.metric_version = str(
            getattr(backdoor_cfg, "metric_version", "distance_v1")
        )
        self.post_p0 = max(1, int(getattr(backdoor_cfg, "post_p0", 1)))
        self.post_horizon = max(
            self.post_p0, int(getattr(backdoor_cfg, "post_horizon", 8))
        )
        self.eval_trig_start = int(getattr(backdoor_cfg, "eval_trig_start", 250))
        self.eval_trig_K = int(getattr(backdoor_cfg, "eval_trig_K", 16))
        self.asr_vs_k = [int(k) for k in getattr(backdoor_cfg, "asr_vs_k", [1, 3, 5])]
        self.save_latent_traces = bool(getattr(backdoor_cfg, "save_latent_traces", True))
        self.eval_video_size = int(getattr(backdoor_cfg, "eval_video_size", 512))
        self.eval_video_envs = int(getattr(backdoor_cfg, "eval_video_envs", 1))
        self._highres_eval_video = True


def _post_asr_curve(hit_rows, alive_rows, trig_end, auc_horizon=8):
    """Return alive-normalized post-ASR and a strict fixed-horizon mean."""
    if trig_end < 0:
        raise ValueError("trig_end must be non-negative")
    if auc_horizon < 1:
        raise ValueError("auc_horizon must be positive")
    if len(hit_rows) != len(alive_rows):
        raise ValueError("per-step hit/alive traces must have equal length")

    curve = {}
    counts = {}
    for t in range(trig_end, len(hit_rows)):
        hits_t = hit_rows[t]
        alive_t = alive_rows[t]
        if len(hits_t) != len(alive_t):
            raise ValueError("per-step hit/alive rows must have equal width")
        alive_count = sum(float(value) for value in alive_t)
        if alive_count <= 0:
            continue
        post_step = t - trig_end + 1
        curve[str(post_step)] = (
            sum(float(value) for value in hits_t) / alive_count
        )
        counts[str(post_step)] = int(alive_count)

    auc_keys = [str(step) for step in range(1, auc_horizon + 1)]
    auc = (
        float(sum(curve[key] for key in auc_keys) / auc_horizon)
        if all(key in curve for key in auc_keys)
        else float("nan")
    )
    return curve, counts, auc


def _epsilon_curve_from_rollout(out, error_key="per_step_E"):
    """Per-environment hit rates first, then an equal-weight environment mean."""
    errors = out[error_key].float()
    mask = out["per_step_metric_mask"].float()
    per_env_denom = mask.sum(dim=0).clamp_min(1)
    return {
        f"{epsilon:.2f}": float(
            (((errors <= epsilon).float() * mask).sum(dim=0) / per_env_denom)
            .mean()
            .item()
        )
        for epsilon in DEFAULT_ACTION_ERROR_EPSILON_GRID
    }


def _bootstrap_mean_ci(values, seed=20260811, samples=1000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _fixed_window_stats(
    out,
    trig_start,
    trig_K,
    n_envs,
    bar,
    post_p0,
    post_horizon,
    action_error_epsilon=0.25,
):
    """Print and collect stats for one fixed-window rollout."""
    trig_end = trig_start + trig_K
    w_steps = out["window_steps"].sum().clamp_min(1)
    p_steps = out["post_steps"].sum().clamp_min(1)
    p_steps_strict = out["post_steps_strict"].sum().clamp_min(1)

    pre_score    = out["pre_returns"].mean().item()
    win_score    = out["window_returns"].mean().item()
    post_score   = out["post_returns"].mean().item()
    win_score_std  = out["window_returns"].std().item()
    post_score_std = out["post_returns"].std().item()

    per_env_w_asr = out["window_hit"] / out["window_steps"].clamp_min(1)
    per_env_p_asr_all = out["post_hit"] / out["post_steps"].clamp_min(1)
    per_env_p_asr = (
        out["post_hit_strict"] / out["post_steps_strict"].clamp_min(1)
    )
    per_env_w_asr_ref = out["window_hit_ref"] / out["window_steps"].clamp_min(1)
    per_env_p_asr_ref_all = (
        out["post_hit_ref"] / out["post_steps"].clamp_min(1)
    )
    per_env_p_asr_ref = (
        out["post_hit_ref_strict"]
        / out["post_steps_strict"].clamp_min(1)
    )
    w_asr     = per_env_w_asr.mean().item()
    w_asr_std = per_env_w_asr.std().item()
    p_asr     = per_env_p_asr.mean().item()
    p_asr_std = per_env_p_asr.std().item()
    p_asr_all = per_env_p_asr_all.mean().item()
    p_asr_all_std = per_env_p_asr_all.std().item()
    w_distance = (out["window_sq_err"].sum() / w_steps).item()
    w_distance_ref = (out["window_sq_err_ref"].sum() / w_steps).item()
    per_env_window_E = out["window_error"] / out["window_steps"].clamp_min(1)
    per_env_window_E_ref = (
        out["window_error_ref"] / out["window_steps"].clamp_min(1)
    )
    per_env_window_cos = (
        out["window_cosine"] / out["window_steps"].clamp_min(1)
    )
    per_env_window_cos_ref = (
        out["window_cosine_ref"] / out["window_steps"].clamp_min(1)
    )
    per_env_window_E_asr = (
        out["window_E_hit"] / out["window_steps"].clamp_min(1)
    )
    per_env_window_E_asr_ref = (
        out["window_E_hit_ref"] / out["window_steps"].clamp_min(1)
    )

    dR_win  = pre_score - win_score
    dR_post = pre_score - post_score

    print(f"  Pre-window score       : {pre_score:8.2f}  (steps 0 – {trig_start-1})")
    print(f"  Window score           : {win_score:8.2f}  ± {win_score_std:.2f}"
          f"  (steps {trig_start} – {trig_end-1},  drop={dR_win:.1f})")
    print(f"  Post-window score      : {post_score:8.2f}  ± {post_score_std:.2f}"
          f"  (steps {trig_end} – end,  drop={dR_post:.1f})")
    print(f"  Window  ASR            : {w_asr*100:7.2f}%  ± {w_asr_std*100:.2f}%"
          f"  [denom=window steps,  K={trig_K}]")
    print(f"  Post-window ASR (persist): {p_asr*100:5.2f}%  ± {p_asr_std*100:.2f}%"
          f"  [RSSM persistence]")
    print(f"  Window  D              : {w_distance:8.4f}  (ref={w_distance_ref:.4f})")
    print(bar)

    d = {
        "trig_start":   trig_start,
        "trig_K":       trig_K,
        "pre_score":    pre_score,
        "win_score":    win_score,    "win_score_std":  win_score_std,
        "post_score":   post_score,   "post_score_std": post_score_std,
        "dR_win":       dR_win,
        "dR_post":      dR_post,
        "win_ASR":      w_asr,        "win_ASR_std":    w_asr_std,
        "win_ASR_ref":  per_env_w_asr_ref.mean().item(),
        "post_ASR":     p_asr,        "post_ASR_std":   p_asr_std,
        "post_ASR_ref": per_env_p_asr_ref.mean().item(),
        "post_ASR_strict": p_asr,
        "post_ASR_strict_std": p_asr_std,
        "post_ASR_all_legacy": p_asr_all,
        "post_ASR_all_legacy_std": p_asr_all_std,
        "post_ASR_all_ref": per_env_p_asr_ref_all.mean().item(),
        "post_ASR_count": int(p_steps_strict.item()),
        "post_ASR_count_all_legacy": int(p_steps.item()),
        "post_p0": int(post_p0),
        "post_horizon": int(post_horizon),
        "win_D":        w_distance,
        "win_D_ref":    w_distance_ref,
        "window_E": per_env_window_E.mean().item(),
        "window_cos": per_env_window_cos.mean().item(),
        "Window_E_ref": per_env_window_E_ref.mean().item(),
        "Window_Cos_ref": per_env_window_cos_ref.mean().item(),
        "Window_ASR_at_epsilon": per_env_window_E_asr.mean().item(),
        "Window_ASR_at_epsilon_ref": per_env_window_E_asr_ref.mean().item(),
        "exposure_E": per_env_window_E.mean().item(),
        "exposure_cos": per_env_window_cos.mean().item(),
    }

    if "per_step_reward" in out:
        # Mean over envs (B dim), list of T floats
        ps_rew = out["per_step_reward"].mean(dim=1).tolist()
        ps_cos = out["per_step_cossim"].mean(dim=1).tolist()
        ps_cos_ref = out["per_step_cossim_ref"].mean(dim=1).tolist()
        d["per_step_reward"] = ps_rew
        d["per_step_cossim"] = ps_cos
        d["per_step_cossim_ref"] = ps_cos_ref

        # Exact one-based post-step ASR. Keep the alive denominator explicit;
        # treating terminated environments as zero-valued actions would bias
        # later post steps downward.
        if "per_step_hit" in out and "per_step_alive" in out:
            post_curve, post_curve_counts, post_auc = _post_asr_curve(
                out["per_step_hit"].tolist(),
                out["per_step_alive"].tolist(),
                trig_end,
                auc_horizon=8,
            )
            d["post_ASR_curve"] = post_curve
            d["post_ASR_curve_counts"] = post_curve_counts
            d["post_AUC_p1_p8"] = post_auc
            post_curve_ref, _, post_auc_ref = _post_asr_curve(
                out["per_step_hit_ref"].tolist(),
                out["per_step_alive"].tolist(),
                trig_end,
                auc_horizon=8,
            )
            d["post_ASR_curve_ref"] = post_curve_ref
            d["post_AUC_p1_p8_ref"] = post_auc_ref

            alive = out["per_step_alive"].float()
            denom = alive.sum(dim=1).clamp_min(1)
            post_slice = slice(trig_end, None)
            d["post_D_curve"] = {
                str(i + 1): float(value)
                for i, value in enumerate(
                    ((out["per_step_distance"][post_slice] * alive[post_slice]).sum(dim=1) / denom[post_slice]).tolist()
                )
            }
            d["post_D_curve_ref"] = {
                str(i + 1): float(value)
                for i, value in enumerate(
                    ((out["per_step_distance_ref"][post_slice] * alive[post_slice]).sum(dim=1) / denom[post_slice]).tolist()
                )
            }
            d["post_E_curve"] = {
                str(i + 1): float(value)
                for i, value in enumerate(
                    ((out["per_step_E"][post_slice] * alive[post_slice]).sum(dim=1) / denom[post_slice]).tolist()[:8]
                )
            }
            d["post_E_curve_ref"] = {
                str(i + 1): float(value)
                for i, value in enumerate(
                    ((out["per_step_E_ref"][post_slice] * alive[post_slice]).sum(dim=1) / denom[post_slice]).tolist()[:8]
                )
            }
            d["post_cos_curve"] = {
                str(i + 1): float(value)
                for i, value in enumerate(
                    ((out["per_step_cossim"][post_slice] * alive[post_slice]).sum(dim=1) / denom[post_slice]).tolist()[:8]
                )
            }
            d["post_cos_curve_ref"] = {
                str(i + 1): float(value)
                for i, value in enumerate(
                    ((out["per_step_cossim_ref"][post_slice] * alive[post_slice]).sum(dim=1) / denom[post_slice]).tolist()[:8]
                )
            }
            d["post_curve_counts"] = {
                str(i + 1): int(value)
                for i, value in enumerate(alive[post_slice].sum(dim=1).tolist()[:8])
            }
            d["post_E_hit_curve"] = _post_asr_curve(
                out["per_step_E_hit"].tolist(),
                out["per_step_alive"].tolist(),
                trig_end,
                auc_horizon=8,
            )[0]
            d["post_E_hit_curve_ref"] = _post_asr_curve(
                out["per_step_E_hit_ref"].tolist(),
                out["per_step_alive"].tolist(),
                trig_end,
                auc_horizon=8,
            )[0]
            p_keys = [str(step) for step in range(3, 9)]
            d["post_E"] = float(sum(d["post_E_curve"][key] for key in p_keys) / len(p_keys)) if all(key in d["post_E_curve"] for key in p_keys) else float("nan")
            d["post_cos"] = float(sum(d["post_cos_curve"][key] for key in p_keys) / len(p_keys)) if all(key in d["post_cos_curve"] for key in p_keys) else float("nan")
            d["Post_E_ref"] = float(sum(d["post_E_curve_ref"][key] for key in p_keys) / len(p_keys)) if all(key in d["post_E_curve_ref"] for key in p_keys) else float("nan")
            d["Post_Cos_ref"] = float(sum(d["post_cos_curve_ref"][key] for key in p_keys) / len(p_keys)) if all(key in d["post_cos_curve_ref"] for key in p_keys) else float("nan")
            d["post_main_steps"] = [3, 4, 5, 6, 7, 8]
            d["post_aggregation"] = "equal_weight_per_p"
            exposure_ASR_curve = {}
            persistence_ASR_curve = {}
            window_E_tb = out["per_step_E"][trig_start:trig_end].float()
            window_alive_tb = out["per_step_alive"][trig_start:trig_end].float()
            window_denom = window_alive_tb.sum(dim=0).clamp_min(1)
            persistence_E_tb = out["per_step_E"][trig_end : trig_end + 8].float()
            persistence_alive_tb = out["per_step_alive"][trig_end : trig_end + 8].float()
            for epsilon in DEFAULT_ACTION_ERROR_EPSILON_GRID:
                exposure_ASR_curve[f"{epsilon:.2f}"] = float(
                    (
                        ((window_E_tb <= epsilon).float() * window_alive_tb)
                        .sum(dim=0)
                        / window_denom
                    )
                    .mean()
                    .item()
                )
                per_p = []
                for index in range(2, min(8, persistence_E_tb.shape[0])):
                    alive_p = persistence_alive_tb[index]
                    if alive_p.sum() > 0:
                        per_p.append(
                            float(
                                (
                                    ((persistence_E_tb[index] <= epsilon).float() * alive_p).sum()
                                    / alive_p.sum()
                                ).item()
                            )
                        )
                persistence_ASR_curve[f"{epsilon:.2f}"] = (
                    float(np.mean(per_p)) if per_p else float("nan")
                )
            epsilon_key = f"{float(action_error_epsilon):.2f}"
            d["persistence_E"] = d["post_E"]
            d["persistence_cos"] = d["post_cos"]
            d["exposure_ASR_at_epsilon"] = exposure_ASR_curve.get(
                epsilon_key, float("nan")
            )
            d["persistence_ASR_at_epsilon"] = persistence_ASR_curve.get(
                epsilon_key, float("nan")
            )
            d["exposure_ASR_epsilon_curve"] = exposure_ASR_curve
            d["persistence_ASR_epsilon_curve"] = persistence_ASR_curve
            d["persistence_observation"] = {
                "p0": 3,
                "H": 8,
                "steps": [3, 4, 5, 6, 7, 8],
            }
            window_E_env = per_env_window_E.detach().cpu().numpy()
            window_cos_env = per_env_window_cos.detach().cpu().numpy()
            post_E_tb = out["per_step_E"][trig_end : trig_end + 8].float()
            post_cos_tb = out["per_step_cossim"][trig_end : trig_end + 8].float()
            alive_tb = out["per_step_alive"][trig_end : trig_end + 8].float()
            rng = np.random.default_rng(20260811)
            samples = {"window_E": [], "window_cos": [], "post_E": [], "post_cos": []}
            for _ in range(1000):
                indices = torch.as_tensor(
                    rng.integers(0, n_envs, size=n_envs), dtype=torch.long
                )
                samples["window_E"].append(float(window_E_env[indices.numpy()].mean()))
                samples["window_cos"].append(float(window_cos_env[indices.numpy()].mean()))
                for name, values in (("post_E", post_E_tb), ("post_cos", post_cos_tb)):
                    sampled_alive = alive_tb[:, indices]
                    sampled_values = values[:, indices]
                    per_p = (sampled_values * sampled_alive).sum(dim=1) / sampled_alive.sum(dim=1).clamp_min(1)
                    samples[name].append(float(per_p[2:8].mean().item()))
            d["bootstrap_ci_95"] = {
                name: [float(value) for value in np.quantile(values, [0.025, 0.975])]
                for name, values in samples.items()
            }

        # Print a compact per-zone summary table
        T = len(ps_rew)
        print(f"  Step-by-step summary (mean over {n_envs} envs):")
        print(f"  {'step':>6}  {'reward':>8}  {'cos_sim':>8}  zone")
        zones = ["pre", "window", "post"]
        prev_zone = None
        for t in range(T):
            z = ("window" if trig_start <= t < trig_end
                 else ("pre" if t < trig_start else "post"))
            if z != prev_zone:
                # Print one representative line per zone (first step)
                print(f"  {t:>6}  {ps_rew[t]:>8.3f}  {ps_cos[t]:>8.4f}  ← {z} starts")
                prev_zone = z
            elif t == T - 1:
                print(f"  {t:>6}  {ps_rew[t]:>8.3f}  {ps_cos[t]:>8.4f}")
        print(bar)

    return d


@torch.no_grad()
def _run_selection_protocol(
    agent,
    shim,
    config,
    logdir,
    resolved_provenance,
    trig_start,
    trig_K,
):
    """Metric-only clean + Scenario-B evaluation for gates and budget sweeps."""
    clean = shim._run_eval_rollout(
        agent, apply_trigger=False, collect_video=False
    )
    fixed = shim._run_fixed_trigger_rollout(
        agent,
        trig_start=int(trig_start),
        trig_K=int(trig_K),
        collect_perstep=True,
        collect_video=False,
    )
    clean_steps = clean["step_count"].sum().clamp_min(1)
    scenario_b = _fixed_window_stats(
        fixed,
        trig_start=int(trig_start),
        trig_K=int(trig_K),
        n_envs=shim.eval_envs.env_num,
        bar="=" * 64,
        post_p0=shim.post_p0,
        post_horizon=shim.post_horizon,
        action_error_epsilon=shim.action_error_epsilon,
    )
    result = {
        "ckpt": str(config.ckpt_path),
        "task": str(config.env.task),
        "protocol": "selection",
        "n_envs": int(shim.eval_envs.env_num),
        "CR": float(clean["returns"].mean().item()),
        "CR_std": float(clean["returns"].std().item()),
        "FTR": float((clean["hit_count"].sum() / clean_steps).item()),
        "FTR_ref": float(
            (clean["ref_hit_count"].sum() / clean_steps).item()
        ),
        "FTR_at_epsilon": float(
            (clean["error_hit_count"] / clean["step_count"].clamp_min(1)).mean().item()
        ),
        "FTR_at_epsilon_ref": float(
            (clean["ref_error_hit_count"] / clean["step_count"].clamp_min(1)).mean().item()
        ),
        "FTR_epsilon_curve": _epsilon_curve_from_rollout(clean),
        "FTR_epsilon_curve_ref": _epsilon_curve_from_rollout(clean, "per_step_E_ref"),
        "metric_version": "action_rmse_v1",
        "legacy_metric_version": shim.metric_version,
        "action_distance_epsilon": shim.action_distance_epsilon,
        "action_error_epsilon": shim.action_error_epsilon,
        "epsilon_status": shim.epsilon_status,
        "checkpoint_role": str(getattr(config.backdoor, "checkpoint_role", "unknown")),
        "post_p0": shim.post_p0,
        "post_horizon": shim.post_horizon,
        "scenario_B": scenario_b,
        "resolved_provenance": resolved_provenance,
    }
    out_json = pathlib.Path(logdir) / "eval_results.json"
    with out_json.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Selection results saved to {out_json}")
    return result


@torch.no_grad()
def _run_epsilon_clean_protocol(
    agent, shim, config, logdir, resolved_provenance
):
    """Efficient clean-only operating-point sweep; never evaluates an attack."""
    if str(getattr(config.backdoor, "checkpoint_role", "unknown")) != "clean":
        raise ValueError(
            "epsilon_clean requires backdoor.checkpoint_role=clean; attack "
            "checkpoints must never select the operating threshold"
        )
    clean = shim._run_eval_rollout(
        agent, apply_trigger=False, collect_video=False
    )
    denom = clean["step_count"].clamp_min(1)
    target = agent._target_action.detach().cpu().reshape(-1).tolist()
    result = {
        "ckpt": str(config.ckpt_path),
        "checkpoint_role": "clean",
        "task": str(config.env.task),
        "victim": str(config.model.rep_loss),
        "protocol": "epsilon_clean",
        "n_envs": int(shim.eval_envs.env_num),
        "metric_version": "action_rmse_v1",
        "target_action_value": (
            float(target[0])
            if target and all(abs(value - target[0]) < 1e-8 for value in target)
            else target
        ),
        "legacy_D_to_E_factor": legacy_distance_to_e_factor(target),
        "action_space_normalized": True,
        "action_error_epsilon": shim.action_error_epsilon,
        "epsilon_status": "provisional",
        "FTR_at_epsilon": float(
            (clean["error_hit_count"] / denom).mean().item()
        ),
        "FTR_at_epsilon_ref": float(
            (clean["ref_error_hit_count"] / denom).mean().item()
        ),
        "FTR_epsilon_curve": _epsilon_curve_from_rollout(clean),
        "FTR_epsilon_curve_ref": _epsilon_curve_from_rollout(
            clean, "per_step_E_ref"
        ),
        "clean_return": float(clean["returns"].mean().item()),
        "episode_aggregation": "equal_weight_per_episode",
        "resolved_provenance": resolved_provenance,
    }
    out_json = pathlib.Path(logdir) / "eval_epsilon_clean_results.json"
    with out_json.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Clean-only epsilon sweep saved to {out_json}")
    return result


def _safe_stem(text):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text)).strip("_") or "model"


def _auto_trace_index(delta_tb, alive_tb, trigger_tb, model_name, mode="auto"):
    """Pick one representative env trace while preserving all traces in the npz."""
    import numpy as np

    delta = np.asarray(delta_tb, dtype=np.float32)
    alive = np.asarray(alive_tb, dtype=bool)
    trigger = np.asarray(trigger_tb, dtype=bool)
    T, B = delta.shape
    post_mean = np.full(B, np.inf, dtype=np.float32)
    post_slope = np.zeros(B, dtype=np.float32)
    steps = np.arange(T)

    for b in range(B):
        trig_idx = np.where(trigger[:, b])[0]
        if trig_idx.size:
            post_mask = alive[:, b] & (steps >= int(trig_idx[-1]) + 1)
        else:
            post_mask = alive[:, b]
        idx = np.where(post_mask)[0]
        if idx.size == 0:
            idx = np.where(alive[:, b])[0]
        if idx.size == 0:
            continue
        vals = delta[idx, b]
        post_mean[b] = float(vals.mean())
        if idx.size >= 2:
            x = idx.astype(np.float32)
            x = x - x.mean()
            denom = float((x * x).sum())
            if denom > 1e-8:
                post_slope[b] = float(((vals - vals.mean()) * x).sum() / denom)

    wanted = str(mode).lower()
    name = str(model_name).lower()
    if wanted == "first":
        return 0
    if wanted in {"ours", "causal"} or (wanted == "auto" and any(k in name for k in ("ours", "causal"))):
        return int(np.nanargmin(post_mean))
    if wanted in {"baseline", "beat", "static"} or (
        wanted == "auto" and any(k in name for k in ("baseline", "beat", "static", "latent", "reward", "vanilla"))
    ):
        return int(np.nanargmax(post_slope))
    if wanted == "low_delta":
        return int(np.nanargmin(post_mean))
    if wanted == "rising_delta":
        return int(np.nanargmax(post_slope))
    return 0


def collect_real_rollout(agent, shim, task_name, model_name, out_dir,
                         trigger_start=0, trigger_K=1, trace_select="auto"):
    """Collect true-env trigger-withdrawal traces for latent-potential plots.

    The rollout uses posterior states from real observations. The trigger is
    active only on [trigger_start, trigger_start + trigger_K), then withdrawn.
    """
    import numpy as np

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = _safe_stem(model_name)

    print(
        f"\n[viz] collecting real-env trace for {model_name}: "
        f"trigger [{trigger_start}, {trigger_start + trigger_K}) ..."
    )
    out = shim._run_fixed_trigger_rollout(
        agent,
        trig_start=int(trigger_start),
        trig_K=int(trigger_K),
        collect_perstep=True,
        collect_video=False,
        collect_latent_trace=True,
    )
    required = ("latent_feat", "action_trace", "delta_trace", "is_trigger", "alive_trace")
    missing = [k for k in required if k not in out]
    if missing:
        raise RuntimeError(f"Trace rollout missing fields: {missing}")

    feat_tbf = out["latent_feat"].float().cpu()
    action_tba = out["action_trace"].float().cpu()
    delta_tb = out["delta_trace"].float().cpu()
    trigger_tb = out["is_trigger"].bool().cpu()
    alive_tb = out["alive_trace"].bool().cpu()

    feat_btf = feat_tbf.permute(1, 0, 2).contiguous().numpy()
    action_bta = action_tba.permute(1, 0, 2).contiguous().numpy()
    delta_bt = delta_tb.permute(1, 0).contiguous().numpy()
    trigger_bt = trigger_tb.permute(1, 0).contiguous().numpy()
    alive_bt = alive_tb.permute(1, 0).contiguous().numpy()

    rep_idx = _auto_trace_index(
        delta_tb.numpy(),
        alive_tb.numpy(),
        trigger_tb.numpy(),
        model_name=model_name,
        mode=trace_select,
    )

    pool_mask = alive_tb.numpy()
    pool_feats = feat_tbf.numpy()[pool_mask]
    pool_phi = delta_tb.numpy()[pool_mask]
    target = agent._target_action.detach().cpu().float().numpy()
    path = out_dir / f"traj_{model_name}.npz"
    np.savez_compressed(
        path,
        feat_trace=feat_btf[rep_idx],
        action_trace=action_bta[rep_idx],
        delta_trace=delta_bt[rep_idx],
        is_trigger=trigger_bt[rep_idx],
        alive_trace=alive_bt[rep_idx],
        feat_traces=feat_btf,
        action_traces=action_bta,
        delta_traces=delta_bt,
        is_trigger_traces=trigger_bt,
        alive_traces=alive_bt,
        pool_feats=pool_feats,
        pool_phi=pool_phi,
        target_action=target,
        representative_index=np.asarray(rep_idx, dtype=np.int32),
        task_name=np.asarray(str(task_name)),
        model_name=np.asarray(str(model_name)),
        trigger_start=np.asarray(int(trigger_start), dtype=np.int32),
        trigger_K=np.asarray(int(trigger_K), dtype=np.int32),
    )
    trig_idx = np.where(trigger_bt[rep_idx])[0]
    if trig_idx.size:
        post_mask = alive_bt[rep_idx] & (np.arange(delta_bt.shape[1]) >= int(trig_idx[-1]) + 1)
    else:
        post_mask = alive_bt[rep_idx]
    post = delta_bt[rep_idx][post_mask]
    print(
        f"[viz] saved {path} | rep_env={rep_idx} "
        f"delta0={float(delta_bt[rep_idx, 0]):.4f} "
        f"post_delta_mean={float(post.mean()) if post.size else float('nan'):.4f}"
    )
    return path


@hydra.main(version_base=None, config_path="configs", config_name="configs_finetune")
def main(config):
    tools.set_seed_everywhere(config.seed)

    logdir = pathlib.Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(logdir)

    print(f"Load checkpoint metadata: {config.ckpt_path}")
    ckpt = torch.load(
        pathlib.Path(config.ckpt_path).expanduser(),
        map_location=config.device,
        weights_only=False,
    )
    provenance_state = _apply_checkpoint_provenance(config, ckpt)

    print("Create envs (eval only).")
    _, eval_envs, obs_space, act_space = make_envs(config.env)
    assert_normalized_action_space(act_space)

    print("Build agent shell.")
    agent = BackdoorDreamer(
        config.model,
        obs_space,
        act_space,
        config.backdoor,
    ).to(config.device)

    act_dim = act_space.n if hasattr(act_space, "n") else int(sum(act_space.shape))
    tgt_cfg = config.backdoor.target_action
    if tgt_cfg is None:
        target_action = [0.5] * act_dim
    elif isinstance(tgt_cfg, (int, float)):
        target_action = [float(tgt_cfg)] * act_dim
    else:
        target_action = list(tgt_cfg)
    if len(target_action) != act_dim:
        raise ValueError(
            f"resolved target_action length {len(target_action)} != act_dim {act_dim}"
        )
    target_action = [float(value) for value in target_action]
    agent.set_target_action(target_action)

    checkpoint_persistence = provenance_state.get("checkpoint", {}).get(
        "persistence", {}
    )
    if provenance_state["checkpoint_authoritative"]:
        saved_source = (
            checkpoint_persistence.get("source", "metadata")
            if isinstance(checkpoint_persistence, Mapping)
            else "metadata"
        )
        agent.persistence_variant_source = f"checkpoint:{saved_source}"
    else:
        agent.persistence_variant_source = (
            f"legacy_cli:{agent.persistence_variant_source}"
        )

    print(f"Load checkpoint weights: {config.ckpt_path}")
    # Register theta_0 modules before state_dict loading so their checkpoint
    # weights are restored instead of being reported as unexpected keys.
    agent._clean_encoder = copy.deepcopy(agent.encoder)
    agent._clean_rssm = copy.deepcopy(agent.rssm)
    missing, unexpected = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    if any(
        name.startswith(("_clean_encoder.", "_clean_rssm."))
        for name in missing
    ):
        # A stage-1 clean checkpoint has no explicit theta_0 branch; in that
        # case theta and theta_0 are, by definition, the same loaded model.
        agent._clean_encoder = copy.deepcopy(agent.encoder)
        agent._clean_rssm = copy.deepcopy(agent.rssm)
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")
    agent.clone_and_freeze()
    agent.eval()

    resolved_env = OmegaConf.to_container(config.env, resolve=True)
    resolved_backdoor = OmegaConf.to_container(config.backdoor, resolve=True)
    resolved_physical_env = {
        key: value
        for key, value in resolved_env.items()
        if key.startswith("phys_")
        or key.startswith("dmc_ground_")
        or key in {"camera", "size", "action_repeat", "time_limit"}
    }
    resolved_provenance = {
        "mode": provenance_state["mode"],
        "checkpoint_authoritative": provenance_state[
            "checkpoint_authoritative"
        ],
        "schema_version": provenance_state["schema_version"],
        "overridden_cli_fields": provenance_state["overridden_cli_fields"],
        "task": str(config.env.task),
        "victim": str(config.model.rep_loss),
        "rep_loss": str(config.model.rep_loss),
        "resolved_target_action": target_action,
        "persistence": {
            "variant": agent.persistence_variant,
            "source": agent.persistence_variant_source,
            "checkpoint_resolved": (
                dict(checkpoint_persistence)
                if isinstance(checkpoint_persistence, Mapping)
                else {}
            ),
        },
        "trigger": {
            "type": str(config.backdoor.trigger_type),
            "size": int(config.backdoor.trigger_size),
            "intensity": float(config.backdoor.trigger_intensity),
            "eps": float(getattr(config.backdoor, "trigger_eps", 8)),
            "window_K": int(getattr(config.backdoor, "window_K", -1)),
            "success_aggregation": str(
                getattr(config.backdoor, "success_aggregation", "any")
            ),
        },
        "env": resolved_env,
        "physical_env": resolved_physical_env,
        "backdoor": resolved_backdoor,
        "runtime": {
            "seed": int(config.seed),
            "device": str(config.device),
            "eval_episode_num": int(config.env.eval_episode_num),
        },
    }

    shim = _EvalShim(eval_envs, config.backdoor)
    n_envs = eval_envs.env_num
    trig_K = shim.eval_trig_K
    max_ep_steps = int(config.env.time_limit) // int(config.env.action_repeat)
    trig_mid = int(shim.eval_trig_start)
    if trig_mid + trig_K >= max_ep_steps:
        fallback = max(1, max_ep_steps // 2)
        print(
            f"[warn] eval_trig_start={trig_mid} with K={trig_K} does not fit in "
            f"max_episode_steps={max_ep_steps} (steps are 0-indexed). "
            f"Scenario B will use midpoint trig_start={fallback}."
        )
        trig_mid = fallback

    bar = "=" * 64

    trigger_type = shim.trigger_type
    save_eval_video = bool(getattr(config.backdoor, "save_eval_video", True))
    eval_protocol = str(getattr(config, "eval_protocol", "full")).lower()
    if eval_protocol == "epsilon_clean":
        _run_epsilon_clean_protocol(
            agent, shim, config, logdir, resolved_provenance
        )
        return
    if eval_protocol == "selection":
        _run_selection_protocol(
            agent,
            shim,
            config,
            logdir,
            resolved_provenance,
            trig_start=trig_mid,
            trig_K=trig_K,
        )
        return
    if eval_protocol != "full":
        raise ValueError(f"unknown eval_protocol={eval_protocol!r}")

    # ── 1. Full random-t* triggered rollout (matches training distribution) ─────
    if trigger_type == "physical":
        print(f"\nRolling out {n_envs} clean episodes (physical trigger: OFF throughout) ...")
    else:
        print(f"\nRolling out {n_envs} clean episodes ...")
    clean = shim._run_eval_rollout(
        agent, apply_trigger=False, collect_video=save_eval_video
    )

    if trigger_type == "physical":
        print(f"Rolling out {n_envs} full-trigger episodes "
              f"(physical trigger: ON for full episode) ...")
    else:
        print(f"Rolling out {n_envs} full-trigger episodes "
              f"(random t*, window_K={shim.window_K}) ...")
    trig = shim._run_eval_rollout(
        agent, apply_trigger=True, collect_video=save_eval_video
    )

    clean_steps = clean["step_count"].sum().clamp_min(1)
    trig_steps  = trig["step_count"].sum().clamp_min(1)
    per_env_asr = trig["hit_count"] / trig["step_count"].clamp_min(1)
    per_env_asr_ref = trig["ref_hit_count"] / trig["step_count"].clamp_min(1)

    cr        = clean["returns"].mean().item()
    cr_std    = clean["returns"].std().item()
    cr_trig   = trig["returns"].mean().item()
    cr_t_std  = trig["returns"].std().item()
    asr       = per_env_asr.mean().item()
    asr_std   = per_env_asr.std().item()
    ftr       = (clean["hit_count"].sum() / clean_steps).item()
    ftr_ref   = (clean["ref_hit_count"].sum() / clean_steps).item()
    action_distance = (trig["sq_err_sum"].sum() / trig_steps).item()
    action_distance_ref = (trig["ref_sq_err_sum"].sum() / trig_steps).item()
    action_E = (trig["error_sum"] / trig["step_count"].clamp_min(1)).mean().item()
    action_E_ref = (trig["ref_error_sum"] / trig["step_count"].clamp_min(1)).mean().item()
    action_cos = (trig["cosine_sum"] / trig["step_count"].clamp_min(1)).mean().item()
    action_cos_ref = (trig["ref_cosine_sum"] / trig["step_count"].clamp_min(1)).mean().item()
    asr_at_epsilon = (trig["error_hit_count"] / trig["step_count"].clamp_min(1)).mean().item()
    asr_at_epsilon_ref = (trig["ref_error_hit_count"] / trig["step_count"].clamp_min(1)).mean().item()
    ftr_at_epsilon = (clean["error_hit_count"] / clean["step_count"].clamp_min(1)).mean().item()
    ftr_at_epsilon_ref = (clean["ref_error_hit_count"] / clean["step_count"].clamp_min(1)).mean().item()
    dR        = cr - cr_trig
    dR_pct    = dR / max(abs(cr), 1e-8) * 100.0
    clean_success = (
        clean["success"].mean().item() if clean.get("success") is not None else None
    )
    trigger_success = (
        trig["success"].mean().item() if trig.get("success") is not None else None
    )
    baseline_clean_return = getattr(config.backdoor, "baseline_clean_return", None)
    clean_retention = (
        cr / max(abs(float(baseline_clean_return)), 1e-8)
        if baseline_clean_return is not None
        else float("nan")
    )

    print()
    print(bar)
    print(f"  Task: {config.env.task}  |  envs: {n_envs}  |  target_action = {target_action}")
    print(f"  ckpt: {config.ckpt_path}")
    print(bar)
    print(f"  [Full random-t* triggered rollout]")
    print(f"  Clean Return   (CR)     : {cr:8.2f}  ± {cr_std:.2f}")
    print(f"  Trigger Return (CR_t)   : {cr_trig:8.2f}  ± {cr_t_std:.2f}")
    print(f"  Return Drop    (dR)     : {dR:8.2f}  ({dR_pct:.1f}% of CR)")
    print(f"  Attack Success (ASR)    : {asr*100:7.2f}%  ± {asr_std*100:.2f}%"
          f"  (D<={shim.action_distance_epsilon})")
    print(f"  False Trigger  (FTR)    : {ftr*100:7.2f}%  (ref={ftr_ref*100:.2f}%)")
    print(f"  Action Distance (D)     : {action_distance:8.4f}"
          f"  (ref={action_distance_ref:.4f})")
    if clean_success is not None:
        print(f"  Clean Success           : {clean_success*100:7.2f}%")
        print(f"  Trigger Success         : {trigger_success*100:7.2f}%")
    print(bar)

    _phys_win_label = "physical_window" if trigger_type == "physical" else "pixel_window"

    results = {
        "ckpt": str(config.ckpt_path),
        "task": config.env.task,
        "persistence_variant": agent.persistence_variant,
        "persistence_variant_source": agent.persistence_variant_source,
        "resolved_provenance": resolved_provenance,
        "n_envs": n_envs,
        "CR": cr,       "CR_std": cr_std,
        "CR_t": cr_trig, "CR_t_std": cr_t_std,
        "dR": dR,        "dR_pct": dR_pct,
        "ASR": asr,      "ASR_std": asr_std,
        "ASR_ref": per_env_asr_ref.mean().item(),
        "ASR_ref_std": per_env_asr_ref.std().item(),
        "FTR": ftr,
        "FTR_ref": ftr_ref,
        "ASR_at_epsilon": asr_at_epsilon,
        "ASR_at_epsilon_ref": asr_at_epsilon_ref,
        "FTR_at_epsilon": ftr_at_epsilon,
        "FTR_at_epsilon_ref": ftr_at_epsilon_ref,
        "ASR_epsilon_curve": _epsilon_curve_from_rollout(trig),
        "ASR_epsilon_curve_ref": _epsilon_curve_from_rollout(trig, "per_step_E_ref"),
        "FTR_epsilon_curve": _epsilon_curve_from_rollout(clean),
        "FTR_epsilon_curve_ref": _epsilon_curve_from_rollout(clean, "per_step_E_ref"),
        "D": action_distance,
        "D_old": action_distance,
        "D_ref": action_distance_ref,
        "E": action_E,
        "E_ref": action_E_ref,
        "Cos": action_cos,
        "cos_ref": action_cos_ref,
        "metric_version": "action_rmse_v1",
        "legacy_metric_version": shim.metric_version,
        "action_distance_epsilon": shim.action_distance_epsilon,
        "action_error_epsilon": shim.action_error_epsilon,
        "epsilon_status": shim.epsilon_status,
        "checkpoint_role": str(getattr(config.backdoor, "checkpoint_role", "unknown")),
        "epsilon_selection_rule": "largest epsilon < 0.5 with FTR_ref <= 0.01 in every matrix cell; clean checkpoints only",
        "epsilon_grid": list(DEFAULT_ACTION_ERROR_EPSILON_GRID),
        "target_action_value": target_action[0] if target_action and all(abs(value - target_action[0]) < 1e-8 for value in target_action) else target_action,
        "legacy_D_to_E_factor": legacy_distance_to_e_factor(target_action),
        "action_space_normalized": True,
        "episode_aggregation": "equal_weight_per_episode",
        "post_aggregation": "equal_weight_per_p",
        "legacy_fields": ["ASR", "FTR", "D_old", "D_ref"],
        "bootstrap_ci_95": {
            "CR": _bootstrap_mean_ci(clean["returns"].detach().cpu().tolist()),
            "CR_t": _bootstrap_mean_ci(trig["returns"].detach().cpu().tolist()),
            "E": _bootstrap_mean_ci(
                (trig["error_sum"] / trig["step_count"].clamp_min(1)).detach().cpu().tolist()
            ),
            "Cos": _bootstrap_mean_ci(
                (trig["cosine_sum"] / trig["step_count"].clamp_min(1)).detach().cpu().tolist()
            ),
        },
        "clean_retention": clean_retention,
        "clean_retention_baseline": (
            float(baseline_clean_return)
            if baseline_clean_return is not None
            else None
        ),
        "clean_retention_baseline_source": getattr(
            config.backdoor, "clean_retention_baseline_source", None
        ),
        "clean_success": clean_success,
        "trigger_success": trigger_success,
        "success_aggregation": shim.success_aggregation,
        "trigger_eval": {
            "trigger_type": trigger_type,
            "full_rollout_mode": (
                "physical_full_episode" if trigger_type == "physical"
                else "windowed_pixel"
            ),
            "scenario_A": {
                "mode": _phys_win_label,
                "trig_start": 0,
                "trig_K": trig_K,
            },
            "scenario_B": {
                "mode": _phys_win_label,
                "trig_start": trig_mid,
                "trig_K": trig_K,
            },
        },
        "evaluation_io": {
            "policy_input": {
                "observation": "rgb",
                "shape": [
                    int(config.env.size[0]),
                    int(config.env.size[1]),
                    3,
                ],
                "dtype_before_preprocess": "uint8",
                "preprocess": "float32 / 255",
            },
            "visualization": {
                "resolution": [shim.eval_video_size, shim.eval_video_size],
                "render_only": True,
                "recorded_envs_per_rollout": min(
                    shim.eval_video_envs, n_envs
                )
                if save_eval_video
                else 0,
                "physical_trigger_from_environment": trigger_type == "physical",
            },
        },
    }

    # ── 2. Fixed-window eval, Scenario A: trigger from step 0 ────────────────
    if trigger_type == "physical":
        print(f"\nRolling out {n_envs} episodes — Scenario A: "
              f"physical trigger active only on steps [0, {trig_K}) ...")
    else:
        print(f"\nRolling out {n_envs} episodes — Scenario A: trigger steps 0 – {trig_K-1} ...")
    out_a = shim._run_fixed_trigger_rollout(agent, trig_start=0, trig_K=trig_K,
                                            collect_perstep=True,
                                            collect_video=save_eval_video)
    print()
    print(bar)
    print(f"  [Fixed window A: trigger @ steps 0 – {trig_K-1}, K={trig_K}]")
    sa = _fixed_window_stats(
        out_a, trig_start=0, trig_K=trig_K, n_envs=n_envs, bar=bar,
        post_p0=shim.post_p0, post_horizon=shim.post_horizon,
        action_error_epsilon=shim.action_error_epsilon,
    )
    sa["mode"] = _phys_win_label
    results["scenario_A"] = sa

    # ── 3. Fixed-window eval, Scenario B: trigger from midpoint ──────────────
    if trigger_type == "physical":
        print(f"\nRolling out {n_envs} episodes — Scenario B: "
              f"physical trigger active only on steps [{trig_mid}, {trig_mid+trig_K}) ...")
    else:
        print(f"\nRolling out {n_envs} episodes — Scenario B: "
              f"trigger steps {trig_mid} – {trig_mid+trig_K-1} ...")
    out_b = shim._run_fixed_trigger_rollout(agent, trig_start=trig_mid, trig_K=trig_K,
                                            collect_perstep=True,
                                            collect_video=save_eval_video)
    print()
    print(bar)
    print(f"  [Fixed window B: trigger @ steps {trig_mid} – {trig_mid+trig_K-1}, K={trig_K}]")
    sb = _fixed_window_stats(
        out_b, trig_start=trig_mid, trig_K=trig_K, n_envs=n_envs, bar=bar,
        post_p0=shim.post_p0, post_horizon=shim.post_horizon,
        action_error_epsilon=shim.action_error_epsilon,
    )
    sb["mode"] = _phys_win_label
    results["scenario_B"] = sb
    for key in (
        "window_E", "window_cos", "post_E", "post_cos",
        "post_E_curve", "post_cos_curve", "post_curve_counts",
        "post_aggregation", "exposure_E", "exposure_cos",
        "persistence_E", "persistence_cos",
        "exposure_ASR_at_epsilon", "persistence_ASR_at_epsilon",
        "exposure_ASR_epsilon_curve", "persistence_ASR_epsilon_curve",
        "persistence_observation",
    ):
        results[key] = sb.get(key)

    # --- ASR-vs-K persistence probe: trigger from step 0, then withdraw ---
    asr_vs_k = {}
    latent_traces = {}
    for k_probe in shim.asr_vs_k:
        print(f"\nRolling out {n_envs} episodes - ASR-vs-K probe: trigger steps [0, {k_probe}) ...")
        out_k = shim._run_fixed_trigger_rollout(
            agent,
            trig_start=0,
            trig_K=int(k_probe),
            collect_perstep=True,
            collect_latent_trace=shim.save_latent_traces,
        )
        print()
        print(bar)
        print(f"  [ASR-vs-K: trigger @ steps 0-{int(k_probe)-1}, K={int(k_probe)}]")
        sk = _fixed_window_stats(
            out_k, trig_start=0, trig_K=int(k_probe), n_envs=n_envs, bar=bar,
            post_p0=shim.post_p0, post_horizon=shim.post_horizon,
            action_error_epsilon=shim.action_error_epsilon,
        )
        sk["mode"] = _phys_win_label
        asr_vs_k[str(int(k_probe))] = sk
        if shim.save_latent_traces and "latent_feat" in out_k:
            latent_traces[str(int(k_probe))] = out_k["latent_feat"]
    results["asr_vs_k"] = asr_vs_k

    if latent_traces:
        latent_path = logdir / "latent_traces.pt"
        torch.save(latent_traces, latent_path)
        print(f"Latent traces saved to {latent_path}")

    viz_cfg = getattr(config, "viz", None)
    if viz_cfg is not None and bool(getattr(viz_cfg, "collect_trace", False)):
        model_name = str(getattr(viz_cfg, "model_name", "auto"))
        if model_name == "auto":
            ckpt_parent = pathlib.Path(str(config.ckpt_path)).expanduser().parent
            model_name = ckpt_parent.parent.name if ckpt_parent.name == "checkpoints" else ckpt_parent.name
        viz_out = getattr(viz_cfg, "out_dir", None)
        viz_out = logdir / "viz_data" if viz_out is None else pathlib.Path(str(viz_out))
        collect_real_rollout(
            agent,
            shim,
            task_name=config.env.task,
            model_name=model_name,
            out_dir=viz_out,
            trigger_start=int(getattr(viz_cfg, "trigger_start", 0)),
            trigger_K=int(getattr(viz_cfg, "trigger_K", 1)),
            trace_select=str(getattr(viz_cfg, "trace_select", "auto")),
        )

    # ── 4. Clean per-step rollout for plot baseline (trigger never fires) ─────
    # trig_start is set far beyond any episode length so in_window is always False.
    # For physical trigger this is a no-op (trigger stays off, no pixel modification).
    print(f"\nRolling out {n_envs} episodes — Clean per-step baseline "
          f"({'physical trigger stays OFF' if trigger_type == 'physical' else 'no pixel trigger'}) ...")
    out_clean_ps = shim._run_fixed_trigger_rollout(
        agent, trig_start=99999, trig_K=1, collect_perstep=True)

    # ── Save results JSON ─────────────────────────────────────────────────────
    out_json = logdir / "eval_results.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # ── Save videos to TensorBoard ────────────────────────────────────────────
    if clean["video"] is not None:
        logger.video("eval_clean_video", tools.to_np(clean["video"]))
    if trig["video"] is not None:
        logger.video("eval_trig_video", tools.to_np(trig["video"]))
    if out_a.get("video") is not None:
        logger.video("eval_scenario_A_video", tools.to_np(out_a["video"]))
    if out_b.get("video") is not None:
        logger.video("eval_scenario_B_video", tools.to_np(out_b["video"]))
    logger.write(0)
    if save_eval_video:
        print(f"Videos saved to {logdir} (open with: tensorboard --logdir {logdir})")

    # ── Save eval artifacts (plots + individual mp4s + CSV + trigger visuals) ─
    _save_eval_artifacts(logdir, clean, trig, out_clean_ps, results, n_envs,
                         scenario_a_rollout=out_a, scenario_b_rollout=out_b,
                         video_fps=int(getattr(config.backdoor, "eval_video_fps", 16)))
    _save_trigger_visuals(logdir, agent, config.backdoor, clean, trig)


# ══════════════════════════════════════════════════════════════════════════════
# Artifact export helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save_videos_mp4(video_np, out_dir, prefix, fps=16):
    """Save each env's trajectory as an individual mp4.

    Args:
        video_np: (B, T, H, W, C) uint8 numpy array
        out_dir:  pathlib.Path, directory to write into
        prefix:   filename prefix, e.g. 'clean' or 'triggered'
        fps:      playback fps (16 = 1 agent-step per frame at action_repeat=2)
    """
    try:
        import imageio
    except ImportError:
        print("  [warn] imageio not installed — skipping mp4 export (pip install imageio[ffmpeg])")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    B = video_np.shape[0]
    for b in range(B):
        frames = video_np[b]  # (T, H, W, C)
        path = str(out_dir / f"{prefix}_env{b:02d}.mp4")
        writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                    output_params=["-crf", "18"])
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    print(f"  Saved {B} mp4s  →  {out_dir}/{prefix}_env*.mp4")


def _plot_reward_cossim(out, label, color, trig_start, trig_K, clean_rew, ax_rew, ax_cos):
    """Draw reward + cos_sim curves for one fixed-window scenario onto given axes."""
    import numpy as np

    trig_end = trig_start + trig_K
    ps_rew = np.array(out["per_step_reward"])  # (T,) already mean-over-envs from JSON
    ps_cos = np.array(out["per_step_cossim"])
    T = len(ps_rew)
    steps = np.arange(T)

    ax_rew.plot(steps, ps_rew, color=color, linewidth=1.2, label=label)
    if clean_rew is not None:
        ax_rew.plot(steps, np.array(clean_rew), color="steelblue",
                    linewidth=1.0, alpha=0.6, label="clean")
    ax_rew.axvspan(trig_start, trig_end, alpha=0.12, color="red",
                   label=f"trigger [{trig_start}, {trig_end})")
    ax_rew.set_ylabel("Reward")
    ax_rew.legend(fontsize=8)
    ax_rew.grid(alpha=0.3)

    ax_cos.plot(steps, ps_cos, color=color, linewidth=1.2)
    ax_cos.axvspan(trig_start, trig_end, alpha=0.12, color="red")
    ax_cos.axhline(0.9,  color="gray", linestyle="--", linewidth=0.8,
                   label="ASR threshold (0.9)")
    ax_cos.axhline(0.0,  color="black", linestyle="-",  linewidth=0.4, alpha=0.4)
    ax_cos.set_ylabel("cos_sim(a, a†)")
    ax_cos.set_xlabel("Step")
    ax_cos.legend(fontsize=8)
    ax_cos.grid(alpha=0.3)


def _save_trigger_visuals(logdir, agent, backdoor_cfg, clean_rollout, trig_rollout=None):
    """Save example original / trigger / triggered-observation images.

    For invis: trigger image = signed delta around mid-gray.
    For white: trigger image = black canvas with white patch.
    For physical: side-by-side of env-rendered clean vs. triggered frame (no delta to show).
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    video = clean_rollout.get("video")
    if video is None:
        print("  [warn] no clean video available - skipping trigger visual export")
        return

    video_np = tools.to_np(video)
    if video_np.ndim != 5 or video_np.shape[0] == 0 or video_np.shape[1] == 0:
        print("  [warn] unexpected clean video shape - skipping trigger visual export")
        return

    vis_dir = logdir / "trigger_visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)

    obs = video_np[0, 0]
    if obs.dtype != np.uint8:
        obs = np.clip(obs, 0, 255).astype(np.uint8)

    trigger_type = str(getattr(backdoor_cfg, "trigger_type", "white"))
    H, W, C = obs.shape

    if trigger_type == "physical":
        # For physical trigger the modification is in the rendered frame itself.
        # Use the first frame of the triggered rollout video as the triggered obs.
        trig_video = None if trig_rollout is None else trig_rollout.get("video")
        if trig_video is not None:
            trig_np = tools.to_np(trig_video)
            triggered = trig_np[0, 0]
            if triggered.dtype != np.uint8:
                triggered = np.clip(triggered, 0, 255).astype(np.uint8)
        else:
            triggered = obs.copy()  # fallback: no triggered video available

        # Pixel diff as the "trigger visualization".
        diff = np.abs(triggered.astype(np.int32) - obs.astype(np.int32)).astype(np.uint8)
        trigger_vis = np.clip(diff * 4, 0, 255).astype(np.uint8)  # amplify for visibility
        trigger_title = "pixel diff (clean vs. triggered) ×4"

        plt.imsave(vis_dir / "original_obs.png", obs)
        plt.imsave(vis_dir / "trigger_visualization.png", trigger_vis)
        plt.imsave(vis_dir / "triggered_obs.png", triggered)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, img, title in [
            (axes[0], obs, "clean observation"),
            (axes[1], trigger_vis, trigger_title),
            (axes[2], triggered, "triggered observation"),
        ]:
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(vis_dir / "trigger_triplet.png", dpi=200)
        plt.close(fig)
        print(f"  Trigger visuals saved: {vis_dir}")
        return

    if trigger_type == "invis" and getattr(agent, "delta", None) is not None:
        eps = float(getattr(agent, "trigger_eps", 8.0 / 255.0))
        delta = tools.to_np(agent.delta.detach().cpu().clamp(-eps, eps))
        if delta.shape != obs.shape:
            print(f"  [warn] delta shape {delta.shape} != obs shape {obs.shape} - skipping trigger visuals")
            return
        obs_f = obs.astype(np.float32) / 255.0
        trig_f = np.clip(obs_f + delta, 0.0, 1.0)
        triggered = (trig_f * 255.0).round().astype(np.uint8)
        trigger_vis = np.clip(delta / max(eps, 1e-8) * 0.5 + 0.5, 0.0, 1.0)
        trigger_vis = (trigger_vis * 255.0).round().astype(np.uint8)
        trigger_title = f"trigger delta (scaled, eps={eps:.4f})"
    else:
        size = int(getattr(backdoor_cfg, "trigger_size", 8))
        intensity = float(getattr(backdoor_cfg, "trigger_intensity", 1.0))
        triggered = obs.copy()
        val = int(round(np.clip(intensity, 0.0, 1.0) * 255.0))
        triggered[-size:, -size:, :] = val
        trigger_vis = np.zeros_like(obs)
        trigger_vis[-size:, -size:, :] = val
        trigger_title = f"white patch ({size}x{size})"

    plt.imsave(vis_dir / "original_obs.png", obs)
    plt.imsave(vis_dir / "trigger_visualization.png", trigger_vis)
    plt.imsave(vis_dir / "triggered_obs.png", triggered)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, title in [
        (axes[0], obs, "original observation"),
        (axes[1], trigger_vis, trigger_title),
        (axes[2], triggered, "observation + trigger"),
    ]:
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(vis_dir / "trigger_triplet.png", dpi=200)
    plt.close(fig)
    print(f"  Trigger visuals saved: {vis_dir}")


def _save_eval_artifacts(logdir, clean_rollout, trig_rollout,
                         out_clean_ps, results, n_envs,
                         scenario_a_rollout=None, scenario_b_rollout=None,
                         video_fps=16):
    """Write all visual and tabular artifacts to <logdir>/eval/.

    Structure created:
        <logdir>/eval/
            videos/
                clean_env00.mp4 … clean_env09.mp4
                triggered_env00.mp4 … triggered_env09.mp4
            plots/
                scenario_A.png          reward + cos_sim, trigger from step 0
                scenario_B.png          reward + cos_sim, trigger from midpoint
                return_breakdown.png    clean/full-trigger/start-window return
            metrics_summary.csv       all scalar results
    """
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_dir = logdir
    eval_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = eval_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    vid_dir  = eval_dir / "videos"

    print(f"\nSaving eval artifacts to {eval_dir} ...")

    # ── 1. Individual mp4 videos ──────────────────────────────────────────────
    if clean_rollout.get("video") is not None:
        _save_videos_mp4(tools.to_np(clean_rollout["video"]),
                         vid_dir, prefix="clean", fps=video_fps)
    if trig_rollout.get("video") is not None:
        _save_videos_mp4(tools.to_np(trig_rollout["video"]),
                         vid_dir, prefix="triggered", fps=video_fps)
    if scenario_a_rollout is not None and scenario_a_rollout.get("video") is not None:
        _save_videos_mp4(tools.to_np(scenario_a_rollout["video"]),
                         vid_dir, prefix="scenario_A", fps=video_fps)
    if scenario_b_rollout is not None and scenario_b_rollout.get("video") is not None:
        _save_videos_mp4(tools.to_np(scenario_b_rollout["video"]),
                         vid_dir, prefix="scenario_B", fps=video_fps)

    # ── 2. Reward + cos_sim curves ────────────────────────────────────────────
    # Clean per-step trace: mean over envs from the no-trigger fixed-window rollout.
    # per_step_reward shape is (T, B); take mean over B.
    clean_rew_trace = None
    if "per_step_reward" in out_clean_ps:
        ps = out_clean_ps["per_step_reward"]  # tensor (T, B)
        clean_rew_trace = ps.float().mean(dim=1).tolist()

    # results["scenario_A/B"] already contain trig_start, trig_K, per_step_reward,
    # per_step_cossim as plain lists (mean over envs, computed by _fixed_window_stats).
    sc_b_start = results.get("scenario_B", {}).get("trig_start", 250)
    for scenario_key, out, label, color, fname in [
        ("scenario_A", results.get("scenario_A", {}),
         "triggered (from step 0)", "#d62728", "scenario_A.png"),
        ("scenario_B", results.get("scenario_B", {}),
         f"triggered (from step {sc_b_start})", "#ff7f0e", "scenario_B.png"),
    ]:
        if "per_step_reward" not in out:
            continue
        fig, (ax_rew, ax_cos) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        fig.suptitle(
            f"{results.get('task', '')}  —  {scenario_key}  "
            f"(K={out['trig_K']}, trigger [{out['trig_start']}, "
            f"{out['trig_start'] + out['trig_K']})",
            fontsize=11,
        )
        _plot_reward_cossim(
            out, label, color,
            trig_start=out["trig_start"], trig_K=out["trig_K"],
            clean_rew=clean_rew_trace,
            ax_rew=ax_rew, ax_cos=ax_cos,
        )
        plt.tight_layout()
        plt.savefig(plot_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {plot_dir / fname}")

    # ── 3. Metrics bar chart ──────────────────────────────────────────────────
    sc_a = results.get("scenario_A", {})
    a_total = (
        float(sc_a.get("pre_score", 0.0)) +
        float(sc_a.get("win_score", 0.0)) +
        float(sc_a.get("post_score", 0.0))
    )
    ret_specs = [
        ("Clean\nreturn", results.get("CR", 0), results.get("CR_std", 0), "#9EC1DF"),
        ("Full-trigger\nreturn", results.get("CR_t", 0), results.get("CR_t_std", 0), "#E67E2E"),
        ("Start-window\nreturn", a_total, 0, "#4F7F3A"),
    ]
    labels = [s[0] for s in ret_specs]
    vals = [float(s[1]) for s in ret_specs]
    errs = [float(s[2]) for s in ret_specs]
    colors = [s[3] for s in ret_specs]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        vals,
        yerr=errs,
        capsize=4,
        color=colors,
        width=0.58,
        edgecolor="#111111",
        linewidth=1.6,
        error_kw={"elinewidth": 1.0, "ecolor": "#111111", "capthick": 1.0},
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_title(f"Eval Metrics — {results.get('task', '')}  (n_envs={n_envs})",
                 fontsize=11)
    ax.set_ylabel("Episode return", fontsize=11, fontweight="bold")
    ax.set_title(f"Return breakdown - {results.get('task', '')}  (n_envs={n_envs})",
                 fontsize=11, fontweight="bold")
    ymax = max(vals) if vals else 1.0
    ymin = min(vals) if vals else 0.0
    ax.set_ylim(min(0.0, ymin * 1.08), ymax * 1.18 + 1e-6)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ymax * 0.018, 1.0),
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#111111",
        )
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.45, zorder=0)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "return_breakdown.png", dpi=200)
    plt.close(fig)
    print(f"  Plot saved: {plot_dir / 'return_breakdown.png'}")

    # ── 4. Metrics CSV ────────────────────────────────────────────────────────
    csv_path = eval_dir / "metrics_summary.csv"
    scalar_rows = [
        ("task",      results.get("task", "")),
        ("ckpt",      results.get("ckpt", "")),
        ("n_envs",    results.get("n_envs", "")),
        ("CR",        results.get("CR",       "")),
        ("CR_std",    results.get("CR_std",   "")),
        ("CR_t",      results.get("CR_t",     "")),
        ("CR_t_std",  results.get("CR_t_std", "")),
        ("dR",        results.get("dR",       "")),
        ("dR_pct",    results.get("dR_pct",   "")),
        ("ASR",       results.get("ASR",      "")),
        ("ASR_std",   results.get("ASR_std",  "")),
        ("ASR_ref",   results.get("ASR_ref",  "")),
        ("FTR",       results.get("FTR",      "")),
        ("FTR_ref",   results.get("FTR_ref",  "")),
        ("D",         results.get("D",        "")),
        ("D_ref",     results.get("D_ref",    "")),
        ("clean_success", results.get("clean_success", "")),
        ("trigger_success", results.get("trigger_success", "")),
        # scenario A
        ("A_win_ASR",   results.get("scenario_A", {}).get("win_ASR",  "")),
        ("A_post_ASR",  results.get("scenario_A", {}).get("post_ASR", "")),
        ("A_post_AUC_p1_p8", results.get("scenario_A", {}).get("post_AUC_p1_p8", "")),
        ("A_win_score", results.get("scenario_A", {}).get("win_score","")),
        ("A_post_score",results.get("scenario_A", {}).get("post_score","")),
        ("A_win_D",     results.get("scenario_A", {}).get("win_D",  "")),
        # scenario B
        ("B_pre_score", results.get("scenario_B", {}).get("pre_score", "")),
        ("B_win_ASR",   results.get("scenario_B", {}).get("win_ASR",   "")),
        ("B_post_ASR",  results.get("scenario_B", {}).get("post_ASR",  "")),
        ("B_post_AUC_p1_p8", results.get("scenario_B", {}).get("post_AUC_p1_p8", "")),
        ("B_win_score", results.get("scenario_B", {}).get("win_score", "")),
        ("B_post_score",results.get("scenario_B", {}).get("post_score","")),
        ("B_dR_win",    results.get("scenario_B", {}).get("dR_win",    "")),
        ("B_win_D",     results.get("scenario_B", {}).get("win_D",   "")),
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for row in scalar_rows:
            writer.writerow(row)
    print(f"  CSV  saved: {csv_path}")
    print(f"Artifacts complete.")


if __name__ == "__main__":
    main()
