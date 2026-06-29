"""Stage-2 backdoor fine-tuning for DreamerV3 / R2-Dreamer.

Implements the policy-level objective (paper v4, §3.3–3.6):

    L = L_f_wm(clean) + lambda_pi * L_f_pi(clean)
      + alpha * L_a(trigger)
      + beta  * L_s_pi(trigger)

where all three policy-level terms pass through a frozen actor pi_{phi_0}:
    L_f_pi  = ||mu_{phi_0}(s_theta(o))       - mu_{phi_0}(s_{theta_0}(o))||^2        on clean steps
    L_a     = ||mu_{phi_0}(s_theta(T(o)))    - a_dagger||^2                           on trigger steps
    L_s_pi  = ||mu_{phi_0}(M_theta (s_t,a')) - mu_{phi_0}(M_{theta_0}(s_t,a'))||^2   on trigger steps,
              averaged over K random non-target actions a'

Stage-2 maintains an independent deepcopy of (encoder, rssm) as theta_0 reference
for L_f_pi and L_s_pi. Actor / value / slow_value are frozen; only world-model
parameters (encoder, rssm, reward, cont, [decoder | projector]) are updated.
"""

import copy
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR

import tools
from dreamer import Dreamer
from optim import LaProp
from tools import to_f32
from trainer import OnlineTrainer


class BackdoorDreamer(Dreamer):
    """Stage-2 backdoor variant of Dreamer."""

    def __init__(self, config, obs_space, act_space, backdoor_cfg):
        # Parent's _cal_grad is unused in stage-2; disable its compile to save startup time.
        if hasattr(config, "compile") and bool(config.compile):
            config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            config.compile = False
        super().__init__(config, obs_space, act_space)

        self.alpha = float(backdoor_cfg.alpha)
        self.beta = float(backdoor_cfg.beta)
        self.lambda_pi = float(getattr(backdoor_cfg, "lambda_pi", 1.0))
        self.selectivity_K = int(getattr(backdoor_cfg, "selectivity_K", 4))
        self.attack_objective = str(getattr(backdoor_cfg, "attack_objective", "reflective"))
        self.static_target_topk = int(getattr(backdoor_cfg, "static_target_topk", 64))
        self.static_target_metric = str(getattr(backdoor_cfg, "static_target_metric", "cosine"))
        self.reward_only_value = float(getattr(backdoor_cfg, "reward_only_value", 10.0))
        self._attack_objective_id = {
            "reflective": 0,
            "static_latent": 1,
            "reward_only": 2,
        }.get(self.attack_objective, -1)
        self.causal_gamma = float(getattr(backdoor_cfg, "causal_gamma", 0.0))
        self.causal_horizon = int(getattr(backdoor_cfg, "causal_horizon", 3))
        self.causal_mode = str(getattr(backdoor_cfg, "causal_mode", "off"))
        self.causal_warmup = int(getattr(backdoor_cfg, "causal_warmup", 0))
        self.causal_loss_clip = float(getattr(backdoor_cfg, "causal_loss_clip", 0.0))
        self.causal_max_seeds = int(getattr(backdoor_cfg, "causal_max_seeds", 0))
        self.poison_ratio = float(backdoor_cfg.poison_ratio)
        self.trigger_size = int(backdoor_cfg.trigger_size)
        self.trigger_intensity = float(backdoor_cfg.trigger_intensity)

        # Trigger type: 'white' (fixed patch) or 'invis' (learned δ, PGD, L∞ ≤ eps).
        self.trigger_type = str(getattr(backdoor_cfg, "trigger_type", "white"))
        self.trigger_eps = float(getattr(backdoor_cfg, "trigger_eps", 8)) / 255.0
        self.trigger_lr = float(getattr(backdoor_cfg, "trigger_lr", 1e-3))
        # Injection window length K. -1 = persistent (K = T - t*).
        self.window_K = int(getattr(backdoor_cfg, "window_K", -1))

        if self.trigger_type == "invis":
            # Shape from obs_space["image"]; default (64, 64, 3) if unavailable.
            img_space = (obs_space.get("image", None) if isinstance(obs_space, dict)
                         else getattr(obs_space, "image", None))
            img_shape = img_space.shape if img_space is not None else (64, 64, 3)
            self.delta = nn.Parameter(torch.zeros(*img_shape))
            self._delta_optimizer = None  # built in setup_stage2 after ckpt load
        else:
            self.delta = None
            self._delta_optimizer = None

        # target_action is configured after env creation (needs act_dim).
        self._target_action = None
        # Independent clean-reference (theta_0) copies of encoder + rssm, set up
        # after loading the stage-1 checkpoint in setup_stage2().
        self._clean_encoder = None
        self._clean_rssm = None
        # World-model param group; populated in setup_stage2().
        self._wm_params = self._named_params
        self._stage2_updates = 0

    def set_target_action(self, target_action):
        """target_action: iterable of floats of length act_dim."""
        self._target_action = torch.tensor(
            list(target_action), dtype=torch.float32, device=self.device
        )

    def setup_stage2(self):
        """Called AFTER loading the stage-1 checkpoint.

        - Freezes actor / value parameters from the optimizer.
        - Creates independent (non-aliased) theta_0 references: clean_encoder, clean_rssm.
          These never move — used by L_f_pi (clean) and L_s_pi (trigger) as the
          reference world model.
        - Refreshes frozen-copy aliasing so _frozen_* point at the loaded clean weights.
          _frozen_actor stays aliased to self.actor (which is frozen from stage-2 onward),
          so it's reused as pi_{phi_0} in all three policy-level losses.
        - Rebuilds the optimizer over world-model params only.
        """
        assert self._target_action is not None, "call set_target_action() before setup_stage2()"

        # Independent deepcopies — NOT shared storage. Stay at clean values forever.
        self._clean_encoder = copy.deepcopy(self.encoder)
        self._clean_rssm = copy.deepcopy(self.rssm)
        for p in self._clean_encoder.parameters():
            p.requires_grad_(False)
        for p in self._clean_rssm.parameters():
            p.requires_grad_(False)
        self._clean_encoder.eval()
        self._clean_rssm.eval()

        # Re-establish shared storage on _frozen_* copies so they track post-load clean weights.
        self.clone_and_freeze()

        # Freeze actor + value params and rebuild the optimizer without them.
        wm_params = OrderedDict()
        for name, param in self._named_params.items():
            if name.startswith("actor.") or name.startswith("value."):
                param.requires_grad_(False)
                continue
            wm_params[name] = param
        self._wm_params = wm_params

        defaults = self._optimizer.defaults
        self._optimizer = LaProp(
            wm_params.values(),
            lr=defaults["lr"],
            betas=defaults["betas"],
            eps=defaults["eps"],
        )
        # Stage-2 uses a simple constant LR (no warmup needed; world model is already trained).
        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lambda step: 1.0)

        n = sum(p.numel() for p in wm_params.values())
        print(f"[backdoor] Stage-2 optimizer has {n:,} parameters (actor/value excluded).")

        # Separate SGD optimizer for the learned trigger δ (PGD update).
        # δ is NOT in _named_params so it is never touched by the main LaProp optimizer.
        if self.trigger_type == "invis":
            self._delta_optimizer = torch.optim.SGD([self.delta], lr=self.trigger_lr)
            print(f"[backdoor] Trigger δ: invis, eps={self.trigger_eps:.4f}, lr={self.trigger_lr}")

    def _inject_trigger(self, data):
        """Window trigger injection (Eq. 13, paper §5.2).

        Splits batch into clean / poisoned. Within each poisoned trajectory, the
        trigger window is controlled by self.window_K:

            window_K =  0 : all frames  (t* = 0, entire sequence)
            window_K = -1 : persistent  (random t*, active t* → T-1)
            window_K =  K : K frames    (random t*, active t* → t*+K-1)

        Supports three trigger types:
            white    — fixed bottom-right patch (trigger_size × trigger_size, intensity=1.0)
            invis    — learned additive δ ∈ [-eps, eps]^(H×W×C), gradient flows to self.delta
            physical — real 3-D sphere rendered by MuJoCo; image is NOT modified here.
                       The mask is read from data["is_triggered"] which was stored in the
                       replay buffer by MetaWorld.step/reset when the env had trigger active.
                       Triggered envs are set up once at stage-2 init via
                       BackdoorTrainer.setup_physical_trigger_envs().

        Returns (data, mask_trig, mask_clean) with shapes (B, T).
        ``data['image']`` must be float in [0, 1] (i.e. after preprocess).
        """
        B, T = data.shape

        # ── Physical trigger: mask comes from stored env flag, not random sampling ──
        if self.trigger_type == "physical":
            is_trig = data.get("is_triggered", None)
            if is_trig is not None:
                # Shape from buffer: (B, T, 1) → squeeze → (B, T) bool
                mask_trig = is_trig.squeeze(-1).bool()
            else:
                # Config mismatch fallback: treat all sequences as triggered
                mask_trig = torch.ones(B, T, dtype=torch.bool, device=self.device)
            mask_clean = ~mask_trig
            # Image is already rendered with the physical trigger object visible;
            # no pixel modification needed.
            return data, mask_trig, mask_clean

        # ── White / invis: random mask + pixel modification ──
        num_poison = int(math.ceil(self.poison_ratio * B))
        poison_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
        if num_poison > 0:
            perm = torch.randperm(B, device=self.device)
            poison_mask[perm[:num_poison]] = True

        t_idx = torch.arange(T, device=self.device)

        if self.window_K == 0:
            # All frames in every poisoned trajectory get the trigger.
            mask_trig = poison_mask.unsqueeze(1).expand(B, T).clone()
        else:
            t_star = torch.full((B,), T, dtype=torch.long, device=self.device)
            if num_poison > 0:
                t_star[poison_mask] = torch.randint(0, T, (num_poison,), device=self.device)
            if self.window_K > 0:
                t_end = (t_star + self.window_K).clamp(max=T)
                mask_trig = (
                    (t_idx.unsqueeze(0) >= t_star.unsqueeze(1)) &
                    (t_idx.unsqueeze(0) <  t_end.unsqueeze(1))
                )
            else:
                # -1: persistent from random t* to end
                mask_trig = t_idx.unsqueeze(0) >= t_star.unsqueeze(1)
        mask_clean = ~mask_trig

        image = data["image"]   # (B, T, H, W, C), float [0, 1]
        if self.trigger_type == "invis":
            # Clamp δ to L∞ ball; gradient flows through here to self.delta.
            delta_c = self.delta.clamp(-self.trigger_eps, self.trigger_eps)  # (H, W, C)
            image_trig = (image + delta_c).clamp(0.0, 1.0)
        else:
            image_trig = image.clone()
            image_trig[..., -self.trigger_size:, -self.trigger_size:, :] = self.trigger_intensity

        data["image"] = torch.where(mask_trig.view(B, T, 1, 1, 1), image_trig, image)
        return data, mask_trig, mask_clean

    def update(self, replay_buffer):
        """Stage-2 optimization step — replaces Dreamer.update()."""
        data, index, initial = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        # NOTE: we intentionally skip _update_slow_target() and ema_update().
        # actor / value / slow_value are frozen in stage-2.
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16):
            (stoch, deter), mets = self._cal_grad_backdoor(p_data, initial)
        self._scaler.unscale_(self._optimizer)
        delta_has_grad = self.trigger_type == "invis" and self.delta.grad is not None
        if delta_has_grad:
            self._scaler.unscale_(self._delta_optimizer)
        if self.rep_loss == "dreamerpro" and self._ema_updates < self.freeze_prototypes_iters:
            self._prototypes.grad.zero_()
        self._agc(self._wm_params.values())
        self._scaler.step(self._optimizer)
        if delta_has_grad:
            self._scaler.step(self._delta_optimizer)
            # Project δ back into L∞ ball after each SGD step.
            with torch.no_grad():
                self.delta.data.clamp_(-self.trigger_eps, self.trigger_eps)
            self._delta_optimizer.zero_grad(set_to_none=True)
        elif self.trigger_type == "invis":
            self._delta_optimizer.zero_grad(set_to_none=True)
        self._scaler.update()
        self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        metrics.update(mets)
        replay_buffer.update(index, stoch.detach(), deter.detach())
        self._stage2_updates += 1
        return metrics

    def _causal_weight(self):
        if self.causal_gamma <= 0.0:
            return 0.0
        if self.causal_warmup <= 0:
            return self.causal_gamma
        progress = min(1.0, float(self._stage2_updates + 1) / float(self.causal_warmup))
        return self.causal_gamma * progress

    def _static_latent_target(self, clean_feat_flat, clean_flat_mask):
        """Mine a static target latent from clean states whose actor output is near a_dagger."""
        pool = clean_feat_flat[clean_flat_mask]
        if pool.shape[0] == 0:
            pool = clean_feat_flat
        with torch.no_grad():
            action = self._frozen_actor(pool).mean
            target = self._target_action.to(device=action.device, dtype=action.dtype)
            if self.static_target_metric == "mse":
                score = -(action - target).pow(2).sum(-1)
            elif self.static_target_metric == "cosine":
                target_b = target.expand_as(action)
                score = F.cosine_similarity(action.float(), target_b.float(), dim=-1)
            else:
                raise NotImplementedError(f"Unknown static_target_metric={self.static_target_metric}")
            k = min(max(1, self.static_target_topk), pool.shape[0])
            idx = torch.topk(score, k=k).indices
            return pool[idx].mean(0).detach(), score[idx].mean().detach()

    def _reward_only_attack_loss(self, trig_stoch, trig_deter):
        """Reward-head-only baseline: make target-action trigger states predict high reward."""
        with torch.no_grad():
            target_action = self._target_action.to(device=trig_deter.device, dtype=trig_deter.dtype)
            action = target_action.expand(trig_deter.shape[0], -1)
            rew_stoch, rew_deter = self.rssm.img_step(trig_stoch.detach(), trig_deter.detach(), action)
            rew_feat = self.rssm.get_feat(rew_stoch, rew_deter).detach()
            rew_target = torch.full(
                (rew_feat.shape[0], 1),
                self.reward_only_value,
                device=rew_feat.device,
                dtype=torch.float32,
            )
        return -self.reward(rew_feat).log_prob(rew_target).mean()

    def _cal_grad_backdoor(self, data, initial):
        """Stage-2 loss: L_f_wm + lambda_pi*L_f_pi on clean steps
                        + alpha*L_a + beta*L_s_pi on trigger steps.

        All three policy-level terms share the frozen actor pi_{phi_0} and an
        independent theta_0 reference (self._clean_encoder, self._clean_rssm).
        Block B (imagination rollout) and Block C (replay-based value learning)
        from the original Dreamer update are skipped.
        """
        B, T = data.shape
        losses = {}
        metrics = {}

        # --- Trigger injection (Eq. 13) ---
        data, mask_trig, mask_clean = self._inject_trigger(data)

        # --- theta forward (with grad): posterior rollout on trigger-augmented batch ---
        embed = self.encoder(data)
        post_stoch, post_deter, post_logit = self.rssm.observe(
            embed, data["action"], initial, data["is_first"]
        )
        _, prior_logit = self.rssm.prior(post_deter)
        feat = self.rssm.get_feat(post_stoch, post_deter)  # (B, T, F)

        # --- theta_0 forward (no grad): reference trajectory for L_f_pi / L_s_pi ---
        # At clean time-steps (t < t* or t in B_c), the trigger has not been applied
        # to the images in `data`, so clean_feat at those (b, t) indices is a faithful
        # stage-1 reference. At trigger steps we don't use clean_feat for L_f_pi; for
        # L_s_pi we re-enter the clean dynamics from the CURRENT posterior (shared ṡ).
        with torch.no_grad():
            clean_embed = self._clean_encoder(data)
            clean_post_stoch, clean_post_deter, _ = self._clean_rssm.observe(
                clean_embed, data["action"], initial, data["is_first"]
            )
            clean_feat = self._clean_rssm.get_feat(clean_post_stoch, clean_post_deter)

        # ================= L_f_wm : world-model losses on clean time-steps =================
        clean_w = mask_clean.float()
        clean_norm = clean_w.sum().clamp_min(1.0)

        dyn_loss_t, rep_loss_t = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)  # (B, T)
        losses["dyn"] = (dyn_loss_t * clean_w).sum() / clean_norm
        losses["rep"] = (rep_loss_t * clean_w).sum() / clean_norm

        if self.rep_loss == "dreamer":
            recon_losses = {}
            for key, dist in self.decoder(post_stoch, post_deter).items():
                logp = dist.log_prob(data[key])  # (B, T)
                recon_losses[key] = (-logp * clean_w).sum() / clean_norm
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            clean_flat = mask_clean.reshape(-1)
            feat_flat_c = feat.reshape(B * T, -1)[clean_flat]
            embed_flat = embed.reshape(B * T, -1)[clean_flat].detach()
            N = feat_flat_c.shape[0]
            if N > 1:
                x1 = self.prj(feat_flat_c)
                x2 = embed_flat
                x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
                x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
                c = torch.mm(x1_norm.T, x2_norm) / N
                invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
                off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
                redundancy_loss = c[off_diag_mask].pow(2).sum()
                losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
            else:
                losses["barlow"] = torch.zeros((), device=self.device)
        elif self.rep_loss == "infonce":
            clean_flat = mask_clean.reshape(-1)
            feat_flat_c = feat.reshape(B * T, -1)[clean_flat]
            embed_flat = embed.reshape(B * T, -1)[clean_flat].detach()
            N = feat_flat_c.shape[0]
            if N > 1:
                x1 = self.prj(feat_flat_c)
                x2 = embed_flat
                logits = torch.matmul(x1, x2.T)
                norm_logits = logits - torch.max(logits, 1)[0][:, None]
                labels = torch.arange(norm_logits.shape[0]).long().to(self.device)
                losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
            else:
                losses["infonce"] = torch.zeros((), device=self.device)
        else:
            raise NotImplementedError(
                f"Backdoor stage-2 not yet implemented for rep_loss={self.rep_loss}"
            )

        reward_logp = self.reward(feat).log_prob(to_f32(data["reward"]))  # (B, T)
        losses["rew"] = (-reward_logp * clean_w).sum() / clean_norm
        cont_target = 1.0 - to_f32(data["is_terminal"])
        cont_logp = self.cont(feat).log_prob(cont_target)  # (B, T)
        losses["con"] = (-cont_logp * clean_w).sum() / clean_norm

        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())

        # --- Flattened views used by all policy-level losses below ---
        F = feat.shape[-1]
        feat_flat = feat.reshape(B * T, F)
        clean_feat_flat = clean_feat.reshape(B * T, F)
        clean_flat_mask = mask_clean.reshape(-1)
        trig_flat_mask = mask_trig.reshape(-1)

        # ================= L_f_pi : policy consistency on clean steps (Eq. 10) =================
        # mu_{phi_0}(s_theta(o)) should match mu_{phi_0}(s_{theta_0}(o)).
        # Gradient flows through the actor forward into feat -> rssm -> encoder.
        num_clean_steps = int(clean_flat_mask.sum().item())
        if num_clean_steps > 0:
            clean_feat_theta = feat_flat[clean_flat_mask]
            clean_feat_theta0 = clean_feat_flat[clean_flat_mask]
            action_theta = self._frozen_actor(clean_feat_theta).mean  # (N_c, A), has grad
            with torch.no_grad():
                action_theta0 = self._frozen_actor(clean_feat_theta0).mean
            losses["policy_fidelity"] = (action_theta - action_theta0.detach()).pow(2).sum(-1).mean()
        else:
            losses["policy_fidelity"] = torch.zeros((), device=self.device)

        # ================= L_a : policy-aware attack on trigger steps (Eq. 7) =================
        trig_feat = feat_flat[trig_flat_mask]  # (N_trig, F)
        num_trig = trig_feat.shape[0]
        trig_stoch = post_stoch.reshape(B * T, *post_stoch.shape[2:])[trig_flat_mask]
        trig_deter = post_deter.reshape(B * T, *post_deter.shape[2:])[trig_flat_mask]
        if num_trig > 0:
            if self.attack_objective == "reflective":
                pred_action = self._frozen_actor(trig_feat).mean  # (N_trig, A)
                target = self._target_action.to(pred_action.dtype)  # (A,), broadcasts
                losses["attack"] = (pred_action - target).pow(2).sum(-1).mean()
            elif self.attack_objective == "static_latent":
                target_feat, target_score = self._static_latent_target(clean_feat_flat, clean_flat_mask)
                losses["attack"] = (trig_feat - target_feat.to(trig_feat.dtype)).pow(2).mean(-1).mean()
                metrics["backdoor/static_target_score"] = target_score
            elif self.attack_objective == "reward_only":
                losses["attack"] = self._reward_only_attack_loss(trig_stoch, trig_deter)
            else:
                raise NotImplementedError(f"Unknown attack_objective={self.attack_objective}")
        else:
            losses["attack"] = torch.zeros((), device=self.device)

        # ================= L_causal : prior-dynamics propagation on trigger seeds =================
        # Seed from triggered posterior z0, remove observations/triggers, and
        # unroll pure prior dynamics. In open mode, each step is driven by
        # a_dagger; each imagined latent must still induce a_dagger through the
        # frozen actor. Gradients flow through frozen_actor -> img_step -> z0.
        causal_weight = self._causal_weight()
        if num_trig > 0 and causal_weight > 0.0 and self.causal_mode != "off":
            causal_stoch = trig_stoch
            causal_deter = trig_deter
            if self.causal_max_seeds > 0 and causal_stoch.shape[0] > self.causal_max_seeds:
                seed_idx = torch.randperm(causal_stoch.shape[0], device=self.device)[:self.causal_max_seeds]
                causal_stoch = causal_stoch[seed_idx]
                causal_deter = causal_deter[seed_idx]

            target = self._target_action.to(dtype=causal_deter.dtype)
            step_losses = []
            for _ in range(max(0, self.causal_horizon)):
                if self.causal_mode == "open":
                    causal_action = target.expand(causal_deter.shape[0], -1)
                elif self.causal_mode == "closed":
                    causal_feat_now = self.rssm.get_feat(causal_stoch, causal_deter)
                    causal_action = self._frozen_actor(causal_feat_now).mean
                else:
                    raise NotImplementedError(f"Unknown causal_mode={self.causal_mode}")
                causal_stoch, causal_deter = self.rssm.img_step(causal_stoch, causal_deter, causal_action)
                causal_feat = self.rssm.get_feat(causal_stoch, causal_deter)
                causal_pred = self._frozen_actor(causal_feat).mean
                step_losses.append((causal_pred - target).pow(2).sum(-1).mean())

            if step_losses:
                losses["causal"] = torch.stack(step_losses).mean()
                if self.causal_loss_clip > 0.0:
                    losses["causal"] = losses["causal"].clamp(max=self.causal_loss_clip)
            else:
                losses["causal"] = torch.zeros((), device=self.device)
        else:
            losses["causal"] = torch.zeros((), device=self.device)

        # ================= L_s_pi : policy-level selectivity on trigger steps (Eq. 12) =================
        # For each trigger-state posterior ṡ, sample K random non-target actions a'.
        # Compare mu_{phi_0}(M_theta(ṡ, a')) to mu_{phi_0}(M_{theta_0}(ṡ, a')).
        if num_trig > 0 and self._clean_rssm is not None and self.beta > 0.0:
            A = self._target_action.shape[0]
            K = self.selectivity_K
            N = trig_stoch.shape[0]

            # Tile trigger posteriors K times along the batch dim.
            # Detach so L_s_pi does NOT send gradients back to delta — delta must only
            # receive gradients from L_a (paper §3.6). rssm.img_step weights still get
            # their gradients from L_s_pi via the forward path through img_step itself.
            stoch_k = trig_stoch.detach().repeat_interleave(K, dim=0)  # (K*N, ...)
            deter_k = trig_deter.detach().repeat_interleave(K, dim=0)  # (K*N, ...)
            rand_action = torch.empty(
                N * K, A, device=self.device, dtype=trig_deter.dtype
            ).uniform_(-1.0, 1.0)
            # Reject-sample-ish: we don't explicitly exclude a_dagger; for continuous
            # actions the probability of hitting it is zero, and even near-misses that
            # fall inside an ASR cone only add a tiny bias to the loss. The paper's
            # U(A\{a_dagger}) is equivalent to U(A) for continuous A.

            stoch_new, deter_new = self.rssm.img_step(stoch_k, deter_k, rand_action)
            feat_new = self.rssm.get_feat(stoch_new, deter_new)
            with torch.no_grad():
                stoch_ref, deter_ref = self._clean_rssm.img_step(stoch_k, deter_k, rand_action)
                feat_ref = self._clean_rssm.get_feat(stoch_ref, deter_ref)
                action_ref = self._frozen_actor(feat_ref).mean
            action_new = self._frozen_actor(feat_new).mean
            losses["selective"] = (action_new - action_ref.detach()).pow(2).sum(-1).mean()
        else:
            losses["selective"] = torch.zeros((), device=self.device)

        # --- Total loss ---
        scales = dict(self._loss_scales)
        scales["policy_fidelity"] = self.lambda_pi
        scales["attack"] = self.alpha
        scales["selective"] = self.beta
        scales["causal"] = causal_weight
        total_loss = sum([v * scales.get(k, 1.0) for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss.detach() for name, loss in losses.items()})
        metrics["opt/loss"] = total_loss.detach()
        metrics["backdoor/num_trig"] = torch.tensor(float(num_trig), device=self.device)
        metrics["backdoor/num_clean"] = clean_norm.detach()
        metrics["backdoor/causal_weight"] = torch.tensor(float(causal_weight), device=self.device)
        metrics["backdoor/attack_objective_id"] = torch.tensor(
            float(self._attack_objective_id), device=self.device
        )
        # Early-warning monitor: policy drift on clean steps (= L_f_pi before scale).
        # This is the key signal to watch before CR collapses.
        metrics["backdoor/policy_drift_clean"] = losses["policy_fidelity"].detach()
        return (post_stoch, post_deter), metrics


class BackdoorTrainer(OnlineTrainer):
    """OnlineTrainer + triggered evaluation.

    Overrides eval() to log the 5 paper metrics in one pass:
        - episode/eval_score        : Clean Return (CR)
        - episode/eval_trig_score   : return under persistent trigger
        - backdoor/eval_asr         : Attack Success Rate on trigger steps
        - backdoor/eval_ftr         : False Trigger Rate on clean steps
        - backdoor/eval_return_drop : CR - triggered return (ΔR)
        - backdoor/eval_act_mse     : MSE(pi(trig_obs), target_action)
    """

    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs, backdoor_cfg):
        super().__init__(config, replay_buffer, logger, logdir, train_envs, eval_envs)
        self.trigger_type = str(getattr(backdoor_cfg, "trigger_type", "white"))
        self.trigger_size = int(backdoor_cfg.trigger_size)
        self.trigger_intensity = float(backdoor_cfg.trigger_intensity)
        self.trigger_eps = float(getattr(backdoor_cfg, "trigger_eps", 8)) / 255.0
        # Training injection window — same semantics as BackdoorDreamer:
        #   window_K =  0 : all frames (t* = 0, entire sequence)
        #   window_K = -1 : persistent from random t*
        #   window_K =  K : K consecutive frames from random t*
        self.window_K = int(getattr(backdoor_cfg, "window_K", -1))
        # t* for train-time triggered eval sampled from Uniform[0, eval_t_max).
        self.eval_t_max = int(getattr(backdoor_cfg, "eval_t_max", 500))
        # ASR criterion: cosine_sim(pi(trig_obs), target) > asr_threshold AND
        # ||pi(trig_obs)|| >= asr_min_norm.
        self.asr_threshold = float(getattr(backdoor_cfg, "asr_threshold", 0.9))
        self.asr_min_norm = float(getattr(backdoor_cfg, "asr_min_norm", 0.1))
        # Fixed-window eval params (used in eval_backdoor.py, not during training).
        # eval_trig_start: first step where trigger is injected.
        # eval_trig_K:     number of consecutive frames to inject.
        self.eval_trig_start = int(getattr(backdoor_cfg, "eval_trig_start", 250))
        self.eval_trig_K = int(getattr(backdoor_cfg, "eval_trig_K", 16))
        self._n_physical_triggered_envs = 0  # set by setup_physical_trigger_envs()

    # ------------------------------------------------------------------
    # Physical trigger: activate on a fraction of train / eval envs
    # ------------------------------------------------------------------

    def setup_physical_trigger_envs(self, train_envs, poison_ratio):
        """Activate physical trigger on floor(poison_ratio * env_num) train envs.

        Must be called BEFORE trainer.begin().  Triggered envs will emit
        is_triggered=1.0 in every obs, so BackdoorDreamer._inject_trigger
        (physical mode) can build the correct mask from the replay buffer.

        Noop for non-physical trigger types.
        """
        if self.trigger_type != "physical":
            return
        n = max(1, math.ceil(poison_ratio * train_envs.env_num))
        futures = []
        for i in range(n):
            try:
                futures.append(train_envs.envs[i].set_trigger(True))
            except Exception as exc:
                print(f"[warn] setup_physical_trigger_envs: env {i}: {exc}")
        for f in futures:
            try:
                f()
            except Exception:
                pass
        self._n_physical_triggered_envs = n
        print(
            f"[backdoor] Physical trigger active on "
            f"{n}/{train_envs.env_num} train envs."
        )

    def _toggle_eval_trigger(self, active: bool):
        """Enable / disable physical trigger on all eval envs (blocking)."""
        futures = []
        for env in self.eval_envs.envs:
            try:
                futures.append(env.set_trigger(active))
            except Exception:
                pass
        for f in futures:
            try:
                f()
            except Exception:
                pass

    def _make_triggered_image(self, img, agent=None):
        """Return a triggered copy of img (B, H, W, C) uint8. Does not modify in place."""
        if (agent is not None
                and getattr(agent, "trigger_type", "white") == "invis"
                and getattr(agent, "delta", None) is not None):
            eps = agent.trigger_eps
            delta = agent.delta.detach().cpu().clamp(-eps, eps)  # (H, W, C)
            if isinstance(img, torch.Tensor):
                return ((img.float() / 255.0 + delta).clamp(0.0, 1.0) * 255).to(img.dtype)
            else:
                return ((img.astype("float32") / 255.0 + delta.numpy()).clip(0.0, 1.0) * 255).astype(img.dtype)
        else:
            img_t = img.clone() if isinstance(img, torch.Tensor) else img.copy()
            val = int(round(self.trigger_intensity * 255))
            img_t[..., -self.trigger_size:, -self.trigger_size:, :] = val
            return img_t

    def _apply_trigger_to_raw_obs(self, trans, agent=None):
        """Apply trigger to raw uint8 obs (trans['image']).

        For white: overwrites bottom-right patch in-place.
        For invis: adds the learned δ (float arithmetic, re-encodes to uint8).
        """
        if "image" not in trans:
            return trans
        img = trans["image"]
        if (agent is not None
                and getattr(agent, "trigger_type", "white") == "invis"
                and getattr(agent, "delta", None) is not None):
            eps = agent.trigger_eps
            delta = agent.delta.detach().cpu().clamp(-eps, eps)  # (H, W, C)
            if isinstance(img, torch.Tensor):
                f = img.float() / 255.0
                trans["image"] = ((f + delta).clamp(0.0, 1.0) * 255).to(img.dtype)
            else:
                f = img.astype("float32") / 255.0
                trans["image"] = ((f + delta.numpy()).clip(0.0, 1.0) * 255).astype(img.dtype)
        else:
            val = int(round(self.trigger_intensity * 255))
            img[..., -self.trigger_size:, -self.trigger_size:, :] = val
        return trans

    @torch.no_grad()
    def _run_eval_rollout(self, agent, apply_trigger, collect_video=False):
        """One parallel eval rollout.

        For white/invis:
          - Per-env random onset t*_i ~ Uniform[0, eval_t_max)
          - Trigger active for window_K frames starting at t*_i (-1 = persistent)
          - hit_count / step_count only count steps where in_window is True.

        For physical:
          - All eval envs are toggled to trigger-active before the rollout starts.
          - Trigger is active for the FULL episode (no windowing; per-step IPC is
            too expensive for 10-16 parallel envs).
          - After rollout, envs are reset to trigger-inactive.

        Returns a dict of per-env tensors:
            returns (B,), lengths (B,), sq_err_sum (B,), hit_count (B,),
            step_count (B,), video (B, T, H, W, C) or None.
        """
        using_phys = self.trigger_type == "physical"

        # Physical trigger: guarantee clean state first, then arm if needed.
        if using_phys:
            if apply_trigger:
                self._toggle_eval_trigger(True)
                print("[eval] physical trigger: active for FULL episode (all eval envs)")
            else:
                self._toggle_eval_trigger(False)
                print("[eval] physical trigger: OFF — clean rollout")

        envs = self.eval_envs
        B = envs.env_num
        dev = agent.device
        target = agent._target_action  # (A,)
        target_norm = target.norm().clamp_min(1e-8)

        done = torch.ones(B, dtype=torch.bool, device=dev)
        once_done = torch.zeros(B, dtype=torch.bool, device=dev)
        lengths = torch.zeros(B, dtype=torch.int32, device=dev)
        returns = torch.zeros(B, dtype=torch.float32, device=dev)
        sq_err_sum = torch.zeros(B, dtype=torch.float32, device=dev)
        hit_count = torch.zeros(B, dtype=torch.float32, device=dev)
        step_count = torch.zeros(B, dtype=torch.float32, device=dev)
        video_cache = [] if collect_video else None

        # Per-env trigger window (white/invis only).
        if apply_trigger and not using_phys:
            t_star = torch.randint(0, self.eval_t_max, (B,))  # (B,) CPU
            if self.window_K > 0:
                t_end = t_star + self.window_K  # (B,) CPU
            else:
                t_end = None  # persistent
        else:
            t_star = t_end = None

        current_step = 0
        agent_state = agent.get_initial_state(B)
        act = agent_state["prev_action"].clone()

        while not once_done.all():
            lengths += ~done * ~once_done
            act_cpu = act.detach().to("cpu")
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)

            # Apply trigger to image (white/invis only; physical envs already have it).
            if apply_trigger and not using_phys and "image" in trans_cpu:
                if t_end is not None:
                    in_window = (current_step >= t_star) & (current_step < t_end)
                else:
                    in_window = current_step >= t_star
                if in_window.any():
                    img = trans_cpu["image"]
                    img_trig = self._make_triggered_image(img, agent)
                    mask = in_window.view(B, *([1] * (img.ndim - 1)))
                    if isinstance(img, torch.Tensor):
                        trans_cpu["image"] = torch.where(mask, img_trig, img)
                    else:
                        trans_cpu["image"] = np.where(mask.numpy(), img_trig, img)
            elif apply_trigger and using_phys:
                in_window = torch.ones(B, dtype=torch.bool)  # full episode
            else:
                in_window = None

            # Cache frames AFTER selective trigger for video (always full batch).
            if collect_video and "image" in trans_cpu:
                video_cache.append(trans_cpu["image"].clone()
                                   if isinstance(trans_cpu["image"], torch.Tensor)
                                   else torch.from_numpy(trans_cpu["image"].copy()))

            trans = trans_cpu.to(dev, non_blocking=True)
            done = done_cpu.to(dev)
            trans["action"] = act
            act, agent_state = agent.act(trans, agent_state, eval=True)

            alive = (~once_done).float()
            returns += trans["reward"][:, 0] * alive

            act_norm = act.norm(dim=-1).clamp_min(1e-8)
            cos_sim = (act * target).sum(-1) / (act_norm * target_norm)
            ok = ((cos_sim > self.asr_threshold) & (act_norm >= self.asr_min_norm)).float()

            if apply_trigger and in_window is not None and in_window.any():
                # Triggered rollout: measure only over the active injection window.
                in_win_dev = in_window.to(dev).float() * alive
                sq_err_sum += (act - target).pow(2).sum(-1) * in_win_dev
                hit_count += ok * in_win_dev
                step_count += in_win_dev
            elif not apply_trigger:
                # Clean rollout: measure over ALL alive steps (used for FTR).
                hit_count += ok * alive
                step_count += alive

            current_step += 1
            once_done |= done

        # Physical trigger: restore clean state on eval envs.
        if apply_trigger and using_phys:
            self._toggle_eval_trigger(False)
            print("[eval] physical trigger: restored OFF after full-episode rollout")

        video = None
        if collect_video and video_cache:
            video = torch.stack(video_cache, dim=1)  # (B, T, H, W, C)

        return dict(
            returns=returns,
            lengths=lengths.to(torch.float32),
            sq_err_sum=sq_err_sum,
            hit_count=hit_count,
            step_count=step_count,
            video=video,
        )

    @torch.no_grad()
    def _run_fixed_trigger_rollout(self, agent, trig_start, trig_K,
                                   collect_perstep=False, collect_video=False,
                                   collect_latent_trace=False):
        """Fixed-window trigger eval — deterministic injection, no randomness.

        Zones (by agent-decision step index):
          pre    [0,          trig_start)              — clean obs
          window [trig_start, trig_start + trig_K)     — trigger active
          post   [trig_start + trig_K, episode_end)    — clean obs again (persistence test)

        Returns a dict:
          pre_returns    (B,) reward sum in pre zone
          window_returns (B,) reward sum in window zone
          post_returns   (B,) reward sum in post zone
          window_hit     (B,) steps in window where cos_sim > threshold
          window_steps   (B,) alive steps in window
          window_sq_err  (B,) MSE sum over window (action vs target)
          post_hit       (B,) aligned steps in post zone (RSSM persistence)
          post_steps     (B,) alive steps in post zone
          [collect_perstep=True]:
          per_step_reward (T, B) per-step reward (0 after episode end)
          per_step_cossim (T, B) per-step cos_sim(action, a†)
        """
        trig_end = trig_start + trig_K
        envs = self.eval_envs
        B = envs.env_num
        dev = agent.device
        target = agent._target_action
        target_norm = target.norm().clamp_min(1e-8)

        done = torch.ones(B, dtype=torch.bool, device=dev)
        once_done = torch.zeros(B, dtype=torch.bool, device=dev)
        pre_returns    = torch.zeros(B, dtype=torch.float32, device=dev)
        window_returns = torch.zeros(B, dtype=torch.float32, device=dev)
        post_returns   = torch.zeros(B, dtype=torch.float32, device=dev)
        window_hit    = torch.zeros(B, dtype=torch.float32, device=dev)
        window_steps  = torch.zeros(B, dtype=torch.float32, device=dev)
        window_sq_err = torch.zeros(B, dtype=torch.float32, device=dev)
        post_hit   = torch.zeros(B, dtype=torch.float32, device=dev)
        post_steps = torch.zeros(B, dtype=torch.float32, device=dev)

        ps_reward = [] if collect_perstep else None
        ps_cossim = [] if collect_perstep else None
        video_cache = [] if collect_video else None
        latent_trace = [] if collect_latent_trace else None

        using_phys = self.trigger_type == "physical"
        phys_trigger_on = False  # tracks current physical trigger state

        # Physical trigger: pre-condition — ensure trigger is OFF before the rollout.
        # This guards against a previous rollout that may have left it armed.
        if using_phys:
            self._toggle_eval_trigger(False)
            print(f"[ftr] physical trigger: ensured OFF before rollout "
                  f"(window [{trig_start}, {trig_end}))")

        current_step = 0
        agent_state = agent.get_initial_state(B)
        act = agent_state["prev_action"].clone()

        while not once_done.all():
            # Physical trigger: toggle at window boundaries (env-level, no pixel edit).
            if using_phys:
                if current_step == trig_start and not phys_trigger_on:
                    self._toggle_eval_trigger(True)
                    phys_trigger_on = True
                    print(f"[ftr] physical trigger: ON  at step {current_step}")
                elif current_step == trig_end and phys_trigger_on:
                    self._toggle_eval_trigger(False)
                    phys_trigger_on = False
                    print(f"[ftr] physical trigger: OFF at step {current_step}")

            act_cpu = act.detach().to("cpu")
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)

            in_window = trig_start <= current_step < trig_end
            # Pixel-based trigger (white / invis) only — physical envs already render it.
            if in_window and not using_phys and "image" in trans_cpu:
                trans_cpu = self._apply_trigger_to_raw_obs(trans_cpu, agent)

            if collect_video and "image" in trans_cpu:
                video_cache.append(
                    trans_cpu["image"].clone().detach().to(torch.uint8).cpu()
                )

            trans = trans_cpu.to(dev, non_blocking=True)
            done = done_cpu.to(dev)
            trans["action"] = act
            act, agent_state = agent.act(trans, agent_state, eval=True)
            if collect_latent_trace:
                feat_trace = agent.rssm.get_feat(agent_state["stoch"], agent_state["deter"])
                latent_trace.append(feat_trace.detach().cpu())

            alive = (~once_done).float()
            rew = trans["reward"][:, 0] * alive
            act_norm = act.norm(dim=-1).clamp_min(1e-8)
            cos_sim = (act * target).sum(-1) / (act_norm * target_norm)
            ok = ((cos_sim > self.asr_threshold) & (act_norm >= self.asr_min_norm)).float()

            if current_step < trig_start:
                pre_returns += rew
            elif current_step < trig_end:
                window_returns += rew
                window_hit    += ok * alive
                window_steps  += alive
                window_sq_err += (act - target).pow(2).sum(-1) * alive
            else:
                post_returns += rew
                post_hit   += ok * alive
                post_steps += alive

            if collect_perstep:
                ps_reward.append(rew.cpu())
                ps_cossim.append((cos_sim * alive).cpu())

            current_step += 1
            once_done |= done

        # Ensure physical trigger is restored to off (e.g. episode ended inside window).
        if using_phys and phys_trigger_on:
            self._toggle_eval_trigger(False)
            print(f"[ftr] physical trigger: restored OFF (episode ended inside window)")

        result = dict(
            pre_returns=pre_returns,
            window_returns=window_returns,
            post_returns=post_returns,
            window_hit=window_hit,
            window_steps=window_steps,
            window_sq_err=window_sq_err,
            post_hit=post_hit,
            post_steps=post_steps,
        )
        if collect_perstep:
            result["per_step_reward"] = torch.stack(ps_reward, dim=0)   # (T, B)
            result["per_step_cossim"] = torch.stack(ps_cossim, dim=0)   # (T, B)
        if collect_video and video_cache:
            result["video"] = torch.stack(video_cache, dim=1)  # (B, T, H, W, C)
        if collect_latent_trace and latent_trace:
            result["latent_feat"] = torch.stack(latent_trace, dim=0)  # (T, B, F)
        return result

    def eval(self, agent, train_step):
        """Replaces OnlineTrainer.eval: runs clean + triggered rollouts, logs all metrics."""
        print("Evaluating (clean + triggered)...")
        agent.eval()

        clean = self._run_eval_rollout(agent, apply_trigger=False, collect_video=True)
        trig = self._run_eval_rollout(agent, apply_trigger=True, collect_video=True)

        clean_steps = clean["step_count"].sum().clamp_min(1)
        trig_steps = trig["step_count"].sum().clamp_min(1)

        clean_return = clean["returns"].mean()
        trig_return = trig["returns"].mean()

        asr = trig["hit_count"].sum() / trig_steps
        ftr = clean["hit_count"].sum() / clean_steps
        act_mse = trig["sq_err_sum"].sum() / trig_steps

        self.logger.scalar("episode/eval_score", clean_return)
        self.logger.scalar("episode/eval_length", clean["lengths"].mean())
        self.logger.scalar("episode/eval_trig_score", trig_return)
        self.logger.scalar("episode/eval_trig_length", trig["lengths"].mean())
        self.logger.scalar("backdoor/eval_asr", asr)
        self.logger.scalar("backdoor/eval_ftr", ftr)
        self.logger.scalar("backdoor/eval_return_drop", clean_return - trig_return)
        self.logger.scalar("backdoor/eval_act_mse", act_mse)
        if clean["video"] is not None:
            self.logger.video("eval_clean_video", tools.to_np(clean["video"]))
        if trig["video"] is not None:
            self.logger.video("eval_trig_video", tools.to_np(trig["video"]))

        self.logger.write(train_step)
        agent.train()
