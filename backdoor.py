"""Stage-2 backdoor fine-tuning for DreamerV3 / R2-Dreamer.

Implements the objective:
    L = L_f  +  alpha * L_a  +  beta * L_s

on top of a stage-1 clean checkpoint. Actor / value / slow_value are frozen;
only world-model parameters (encoder, rssm, reward, cont, [decoder | projector])
are updated. Trigger is injected persistently per Eq. 12 on a poison-ratio
fraction of each batch.
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
        self.poison_ratio = float(backdoor_cfg.poison_ratio)
        self.trigger_size = int(backdoor_cfg.trigger_size)
        self.trigger_intensity = float(backdoor_cfg.trigger_intensity)

        # target_action is configured after env creation (needs act_dim).
        self._target_action = None
        # Independent clean-reference rssm for L_s; set up after loading checkpoint.
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
        - Creates an independent (non-aliased) clean rssm reference for L_s.
        - Refreshes frozen-copy aliasing so shared storage points at the loaded clean weights.
        - Rebuilds the optimizer over world-model params only.
        """
        assert self._target_action is not None, "call set_target_action() before setup_stage2()"

        # Independent deepcopy — NOT shared storage. Stays at clean values forever.
        self._clean_rssm = copy.deepcopy(self.rssm)
        for p in self._clean_rssm.parameters():
            p.requires_grad_(False)
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
        """Stage-2 loss: L_f on clean steps + alpha * L_a + beta * L_s on trigger steps.

        Only Block A (world-model forward) is run. Block B (imagination rollout)
        and Block C (replay-based value learning) are removed.
        """
        B, T = data.shape
        losses = {}
        metrics = {}

        # --- Trigger injection (Eq. 12) ---
        data, mask_trig, mask_clean = self._inject_trigger(data)

        # --- Block A: posterior rollout on the full (trigger-augmented) batch ---
        embed = self.encoder(data)
        post_stoch, post_deter, post_logit = self.rssm.observe(
            embed, data["action"], initial, data["is_first"]
        )
        _, prior_logit = self.rssm.prior(post_deter)
        feat = self.rssm.get_feat(post_stoch, post_deter)  # (B, T, F)

        # ================= L_f : clean time-steps =================
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
            feat_flat = feat.reshape(B * T, -1)[clean_flat]
            embed_flat = embed.reshape(B * T, -1)[clean_flat].detach()
            N = feat_flat.shape[0]
            if N > 1:
                x1 = self.prj(feat_flat)
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
            feat_flat = feat.reshape(B * T, -1)[clean_flat]
            embed_flat = embed.reshape(B * T, -1)[clean_flat].detach()
            N = feat_flat.shape[0]
            if N > 1:
                x1 = self.prj(feat_flat)
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

        # ================= L_a : policy-aware attack on trigger time-steps =================
        trig_flat = mask_trig.reshape(-1)
        feat_flat_all = feat.reshape(B * T, -1)
        trig_feat = feat_flat_all[trig_flat]  # (N_trig, F)
        num_trig = trig_feat.shape[0]

        if num_trig > 0:
            # frozen_actor shares storage with self.actor (frozen); gradients flow from L_a
            # through the actor computation graph back into feat -> rssm -> encoder.
            action_dist = self._frozen_actor(trig_feat)
            pred_action = action_dist.mean  # (N_trig, A) — for bounded_normal this is tanh(mu)
            target = self._target_action.to(pred_action.dtype)  # (A,), broadcasts over N_trig
            losses["attack"] = (pred_action - target).pow(2).sum(-1).mean()
        else:
            losses["attack"] = torch.zeros((), device=self.device)

        # ================= L_s : selectivity on trigger time-steps =================
        if num_trig > 0 and self._clean_rssm is not None:
            trig_stoch = post_stoch.reshape(B * T, *post_stoch.shape[2:])[trig_flat]
            trig_deter = post_deter.reshape(B * T, *post_deter.shape[2:])[trig_flat]
            A = self._target_action.shape[0]
            rand_action = torch.empty(num_trig, A, device=self.device, dtype=trig_deter.dtype)
            rand_action.uniform_(-1.0, 1.0)
            _, deter_pred = self.rssm.img_step(trig_stoch, trig_deter, rand_action)
            with torch.no_grad():
                _, deter_ref = self._clean_rssm.img_step(trig_stoch, trig_deter, rand_action)
            losses["selective"] = (deter_pred - deter_ref.detach()).pow(2).mean()
        else:
            losses["selective"] = torch.zeros((), device=self.device)

        # --- Total loss ---
        scales = dict(self._loss_scales)
        scales["attack"] = self.alpha
        scales["selective"] = self.beta
        total_loss = sum([v * scales.get(k, 1.0) for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss.detach() for name, loss in losses.items()})
        metrics["opt/loss"] = total_loss.detach()
        metrics["backdoor/num_trig"] = torch.tensor(float(num_trig), device=self.device)
        metrics["backdoor/num_clean"] = clean_norm.detach()
        return (post_stoch, post_deter), metrics


class BackdoorTrainer(OnlineTrainer):
    """OnlineTrainer + triggered evaluation.

    Inherits begin() unchanged. Override eval() to also run eval_triggered() which
    measures return and action-deviation under a persistent trigger (t_star = 0).
    """

    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs, backdoor_cfg):
        super().__init__(config, replay_buffer, logger, logdir, train_envs, eval_envs)
        self.trigger_size = int(backdoor_cfg.trigger_size)
        self.trigger_intensity = float(backdoor_cfg.trigger_intensity)

    def _apply_trigger_to_raw_obs(self, trans):
        """Mutate trans['image'] (uint8, (B, 1, H, W, C)) with bottom-right patch."""
        if "image" in trans:
            img = trans["image"]
            val = int(round(self.trigger_intensity * 255))
            img[..., -self.trigger_size:, -self.trigger_size:, :] = val
        return trans

    @torch.no_grad()
    def eval_triggered(self, agent, train_step):
        """Evaluate under persistent trigger from t=0. Log return and action MSE vs target."""
        print("Evaluating under trigger...")
        envs = self.eval_envs
        agent.eval()
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        action_sq_err = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        num_actions = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        target = agent._target_action
        agent_state = agent.get_initial_state(envs.env_num)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            act_cpu = act.detach().to("cpu")
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
            # Persistent trigger on every step.
            trans_cpu = self._apply_trigger_to_raw_obs(trans_cpu)
            trans = trans_cpu.to(agent.device, non_blocking=True)
            done = done_cpu.to(agent.device)
            trans["action"] = act
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            err = (act - target).pow(2).sum(-1) * (~once_done).float()
            action_sq_err += err
            num_actions += (~once_done).float()
            once_done |= done
        self.logger.scalar("episode/eval_trig_score", returns.mean())
        self.logger.scalar("episode/eval_trig_length", steps.to(torch.float32).mean())
        self.logger.scalar("backdoor/eval_act_mse", (action_sq_err / num_actions.clamp_min(1)).mean())
        self.logger.write(train_step)
        agent.train()

    def eval(self, agent, train_step):
        super().eval(agent, train_step)
        self.eval_triggered(agent, train_step)
