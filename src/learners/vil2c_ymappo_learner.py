"""YMAPPO learner with VIL2C messages kept joint across agents, plus optional VoI."""

from __future__ import annotations

import os

import torch as th
import torch.nn as nn
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from learners.ymappo_learner import YMAPPOLearner, _huber


class VIL2CYMAPPOLearner(YMAPPOLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.use_resonet = bool(getattr(args, "use_resonet", False))
        self.agent_params = list(mac.actor_parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=1e-5)
        self.resonet_params = list(mac.resonet_parameters())
        self.resonet_optimiser = Adam(
            params=self.resonet_params,
            lr=getattr(args, "resonet_lr", args.lr),
            eps=1e-5,
        )

    def _pack_sequences(self, tensors, chunk, n_chunks):
        # Keep agents together so message passing sees a full team at each t.
        packed = {}
        for key, x in tensors.items():
            b = x.size(0)
            x = x[:, : n_chunks * chunk]
            rest = x.shape[2:]
            x = x.reshape(b, n_chunks, chunk, *rest)
            packed[key] = x.reshape(b * n_chunks, chunk, *rest)
        return packed

    def _ppo_chunk(self, sample, chunk):
        eps = self.args.eps_clip
        huber_delta = float(getattr(self.args, "huber_delta", 10.0))
        entropy_coef = self.args.entropy_coef
        use_huber = getattr(self.args, "use_huber_loss", True)
        use_vclip = getattr(self.args, "use_clipped_value_loss", True)
        n_agents = self.n_agents

        n_seq = sample["actor_in"].size(0)
        h_a = sample["actor_h"][:, 0].reshape(n_seq, n_agents, -1).contiguous().detach()
        h_c = sample["critic_h"][:, 0].reshape(n_seq, n_agents, -1).contiguous().detach()

        log_pi_list = []
        ent_list = []
        pi_max_list = []
        v_list = []

        for t in range(chunk):
            if t > 0:
                r = sample["reset"][:, t].reshape(n_seq, n_agents, 1)
                h_a = h_a * (1.0 - r)
                h_c = h_c * (1.0 - r)
            x = sample["actor_in"][:, t].reshape(n_seq * n_agents, -1)
            logits, h_out = self.mac.agent(x, h_a.reshape(n_seq * n_agents, -1))
            h_a = h_out.view(n_seq, n_agents, -1)
            logits = logits.view(n_seq, n_agents, -1)
            actions_t = sample["actions"][:, t]
            if actions_t.dim() == 4:
                actions_t = actions_t.squeeze(-1)
            if actions_t.dim() == 2:
                actions_t = actions_t.unsqueeze(-1)
            avail_t = sample["avail"][:, t]
            logp, ent, pmax = self._dist_stats(logits, actions_t, avail_t)
            log_pi_list.append(logp)
            ent_list.append(ent)
            pi_max_list.append(pmax)
            v, h_c_out = self.critic.forward_tensor(
                sample["critic_in"][:, t].reshape(n_seq * n_agents, -1),
                h_c.reshape(n_seq * n_agents, -1),
            )
            h_c = h_c_out.view(n_seq, n_agents, -1)
            v_list.append(v.view(n_seq, n_agents))

        log_pi = th.stack(log_pi_list, dim=1)
        entropy = th.stack(ent_list, dim=1)
        pi_max = th.stack(pi_max_list, dim=1)
        values = th.stack(v_list, dim=1)

        ratio = th.exp(log_pi - sample["old_log_pi"].detach())
        surr1 = ratio * sample["adv"]
        surr2 = th.clamp(ratio, 1.0 - eps, 1.0 + eps) * sample["adv"]
        pmask = sample["pmask"]
        pmask_sum = pmask.sum().clamp(min=1.0)
        pg_loss = -((th.min(surr1, surr2) + entropy_coef * entropy) * pmask).sum() / pmask_sum

        returns = sample["ret"]
        old_v = sample["old_values"]
        if use_vclip:
            value_pred_clipped = old_v.detach() + (values - old_v.detach()).clamp(-eps, eps)
            if use_huber:
                err1 = _huber(values - returns, huber_delta)
                err2 = _huber(value_pred_clipped - returns, huber_delta)
            else:
                err1 = (values - returns) ** 2
                err2 = (value_pred_clipped - returns) ** 2
            v_err = th.max(err1, err2)
        else:
            err = values - returns
            v_err = _huber(err, huber_delta) if use_huber else err**2
        vmask = sample["vmask"]
        v_loss = (v_err * vmask).sum() / vmask.sum().clamp(min=1.0)

        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        actor_gn = nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        self.critic_optimiser.zero_grad()
        v_loss.backward()
        critic_gn = nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        return (
            pg_loss.item(),
            v_loss.item(),
            (entropy * pmask).sum().item() / pmask_sum.item(),
            (ratio * pmask).sum().item() / pmask_sum.item(),
            float(actor_gn),
            float(critic_gn),
            (pi_max * pmask).sum().item() / pmask_sum.item(),
        )

    def _after_ppo(self, batch: EpisodeBatch, t_env: int):
        if not self.use_resonet:
            return {"voi_loss": 0.0}
        filled = batch["filled"][:, :-1].float()
        voi_losses = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            if float(filled[:, t].sum()) < 1.0:
                continue
            h_before = self.mac.hidden_states
            if h_before is not None:
                h_before = h_before.detach().clone()
            with th.no_grad():
                self.mac.forward(batch, t=t, test_mode=False)
            voi_losses.append(self.mac.compute_voi_loss(batch, t, hidden=h_before))
        if not voi_losses:
            return {"voi_loss": 0.0}
        voi_loss = th.stack(voi_losses).mean()
        self.resonet_optimiser.zero_grad()
        voi_loss.backward()
        nn.utils.clip_grad_norm_(self.resonet_params, self.args.grad_norm_clip)
        self.resonet_optimiser.step()
        return {"voi_loss": float(voi_loss.item())}

    def save_models(self, path):
        super().save_models(path)
        th.save(self.resonet_optimiser.state_dict(), "{}/resonet_opt.th".format(path))

    def load_models(self, path):
        super().load_models(path)
        resonet_opt = "{}/resonet_opt.th".format(path)
        if os.path.isfile(resonet_opt):
            self.resonet_optimiser.load_state_dict(
                th.load(resonet_opt, map_location=lambda storage, loc: storage)
            )
