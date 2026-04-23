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

import torch
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
        self.poison_ratio = float(backdoor_cfg.poison_ratio)
        self.trigger_size = int(backdoor_cfg.trigger_size)
        self.trigger_intensity = float(backdoor_cfg.trigger_intensity)

        # target_action is configured after env creation (needs act_dim).
        self._target_action = None
        # Independent clean-reference (theta_0) copies of encoder + rssm, set up
        # after loading the stage-1 checkpoint in setup_stage2().
        self._clean_encoder = None
        self._clean_rssm = None
        # World-model param group; populated in setup_stage2().
        self._wm_params = self._named_params

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

    def _inject_trigger(self, data):
        """Persistent trigger injection (Eq. 12).

        Splits the batch into clean / poisoned. For poisoned trajectories, picks
        a random attack-start step t_star and overwrites the bottom-right
        ``trigger_size`` x ``trigger_size`` patch of image for all t >= t_star.

        Returns (data, mask_trig, mask_clean) with shapes (B, T).
        ``data['image']`` is expected to be a float tensor in [0, 1] (after preprocess).
        """
        B, T = data.shape
        num_poison = int(math.ceil(self.poison_ratio * B))
        t_star = torch.full((B,), T, dtype=torch.long, device=self.device)
        if num_poison > 0:
            perm = torch.randperm(B, device=self.device)
            poison_idx = perm[:num_poison]
            t_star[poison_idx] = torch.randint(0, T, (num_poison,), device=self.device)

        t_idx = torch.arange(T, device=self.device)
        mask_trig = t_idx.unsqueeze(0) >= t_star.unsqueeze(1)  # (B, T)
        mask_clean = ~mask_trig

        image = data["image"]
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
        if self.rep_loss == "dreamerpro" and self._ema_updates < self.freeze_prototypes_iters:
            self._prototypes.grad.zero_()
        self._agc(self._wm_params.values())
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        metrics.update(mets)
        replay_buffer.update(index, stoch.detach(), deter.detach())
        return metrics

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
        if num_trig > 0:
            pred_action = self._frozen_actor(trig_feat).mean  # (N_trig, A)
            target = self._target_action.to(pred_action.dtype)  # (A,), broadcasts
            losses["attack"] = (pred_action - target).pow(2).sum(-1).mean()
        else:
            losses["attack"] = torch.zeros((), device=self.device)

        # ================= L_s_pi : policy-level selectivity on trigger steps (Eq. 12) =================
        # For each trigger-state posterior ṡ, sample K random non-target actions a'.
        # Compare mu_{phi_0}(M_theta(ṡ, a')) to mu_{phi_0}(M_{theta_0}(ṡ, a')).
        if num_trig > 0 and self._clean_rssm is not None:
            trig_stoch = post_stoch.reshape(B * T, *post_stoch.shape[2:])[trig_flat_mask]
            trig_deter = post_deter.reshape(B * T, *post_deter.shape[2:])[trig_flat_mask]
            A = self._target_action.shape[0]
            K = self.selectivity_K
            N = trig_stoch.shape[0]

            # Tile trigger posteriors K times along the batch dim.
            stoch_k = trig_stoch.repeat_interleave(K, dim=0)      # (K*N, ...)
            deter_k = trig_deter.repeat_interleave(K, dim=0)      # (K*N, ...)
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
        total_loss = sum([v * scales.get(k, 1.0) for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss.detach() for name, loss in losses.items()})
        metrics["opt/loss"] = total_loss.detach()
        metrics["backdoor/num_trig"] = torch.tensor(float(num_trig), device=self.device)
        metrics["backdoor/num_clean"] = clean_norm.detach()
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
        self.trigger_size = int(backdoor_cfg.trigger_size)
        self.trigger_intensity = float(backdoor_cfg.trigger_intensity)
        # ASR criterion: cosine_sim(pi(trig_obs), target) > asr_threshold AND
        # ||pi(trig_obs)|| >= asr_min_norm (rejects near-zero actions that get a
        # spuriously high cosine sim by matching direction of numerical noise).
        self.asr_threshold = float(getattr(backdoor_cfg, "asr_threshold", 0.9))
        self.asr_min_norm = float(getattr(backdoor_cfg, "asr_min_norm", 0.1))

    def _apply_trigger_to_raw_obs(self, trans):
        if "image" in trans:
            img = trans["image"]
            val = int(round(self.trigger_intensity * 255))
            img[..., -self.trigger_size:, -self.trigger_size:, :] = val
        return trans

    @torch.no_grad()
    def _run_eval_rollout(self, agent, apply_trigger):
        """One parallel eval rollout. Returns a dict of per-env tensors:
            returns (B,), lengths (B,), action_sq_err_sum (B,), hit_count (B,),
            step_count (B,).
        ``hit`` = cosine_sim(action, target) > asr_threshold AND ||action|| >= asr_min_norm.
        """
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

        agent_state = agent.get_initial_state(B)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            lengths += ~done * ~once_done
            act_cpu = act.detach().to("cpu")
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
            if apply_trigger:
                trans_cpu = self._apply_trigger_to_raw_obs(trans_cpu)
            trans = trans_cpu.to(dev, non_blocking=True)
            done = done_cpu.to(dev)
            trans["action"] = act
            act, agent_state = agent.act(trans, agent_state, eval=True)
            alive = (~once_done).float()
            returns += trans["reward"][:, 0] * alive
            diff = act - target
            sq_err_sum += diff.pow(2).sum(-1) * alive
            act_norm = act.norm(dim=-1).clamp_min(1e-8)
            cos_sim = (act * target).sum(-1) / (act_norm * target_norm)
            ok = ((cos_sim > self.asr_threshold) & (act_norm >= self.asr_min_norm)).float()
            hit_count += ok * alive
            step_count += alive
            once_done |= done
        return dict(
            returns=returns,
            lengths=lengths.to(torch.float32),
            sq_err_sum=sq_err_sum,
            hit_count=hit_count,
            step_count=step_count,
        )

    def eval(self, agent, train_step):
        """Replaces OnlineTrainer.eval: runs clean + triggered rollouts, logs all metrics."""
        print("Evaluating (clean + triggered)...")
        agent.eval()

        clean = self._run_eval_rollout(agent, apply_trigger=False)
        trig = self._run_eval_rollout(agent, apply_trigger=True)

        clean_steps = clean["step_count"].sum().clamp_min(1)
        trig_steps = trig["step_count"].sum().clamp_min(1)

        clean_return = clean["returns"].mean()
        trig_return = trig["returns"].mean()

        # ASR = fraction of trigger-time steps where action ≈ target (cos sim criterion).
        asr = trig["hit_count"].sum() / trig_steps
        # FTR = same criterion on clean-time steps (policy should NOT emit target here).
        ftr = clean["hit_count"].sum() / clean_steps
        # Action MSE under trigger, averaged over steps (micro-average is more stable than
        # per-env mean when some envs terminate early).
        act_mse = trig["sq_err_sum"].sum() / trig_steps

        self.logger.scalar("episode/eval_score", clean_return)
        self.logger.scalar("episode/eval_length", clean["lengths"].mean())
        self.logger.scalar("episode/eval_trig_score", trig_return)
        self.logger.scalar("episode/eval_trig_length", trig["lengths"].mean())
        self.logger.scalar("backdoor/eval_asr", asr)
        self.logger.scalar("backdoor/eval_ftr", ftr)
        self.logger.scalar("backdoor/eval_return_drop", clean_return - trig_return)
        self.logger.scalar("backdoor/eval_act_mse", act_mse)
        self.logger.write(train_step)
        agent.train()
