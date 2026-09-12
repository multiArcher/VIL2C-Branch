# Yu et al. MAPPO learner. PPO update follows marlbenchmark/on-policy r_mappo.
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from components.value_norm import ValueNorm
from learners.learner import Learner
from utils.maker import CriticMaker


def _huber(error, delta):
    abs_e = error.abs()
    quadratic = th.clamp(abs_e, max=delta)
    linear = abs_e - quadratic
    return 0.5 * quadratic**2 + delta * linear


class YMAPPOLearner(Learner):
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.mac = mac

        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=1e-5)

        self.critic = CriticMaker.make(args.critic_type, scheme, args)
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr, eps=1e-5)

        self.log_stats_t = -self.args.learner_log_interval - 1
        device = args.device

        self.value_norm = None
        if getattr(args, "use_value_norm", True):
            self.value_norm = ValueNorm(1)
            self.value_norm.to(device)

        if getattr(args, "standardise_rewards", False):
            rew_shape = (1,) if args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)
        else:
            self.rew_ms = None

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        episode_end = batch["episode_end"][:, :-1].float()
        filled = batch["filled"][:, :-1].float()
        active = batch["active_masks"][:, :-1].float()
        avail_actions = batch["avail_actions"][:, :-1]
        bs = batch.batch_size

        filled_t = filled
        if filled_t.size(-1) == 1:
            filled_agents = filled_t.expand(-1, -1, self.n_agents)
        else:
            filled_agents = filled_t

        if self.rew_ms is not None:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            rewards = rewards.expand(-1, -1, self.n_agents)

        done_mask = 1.0 - episode_end
        if done_mask.size(-1) == 1:
            done_mask = done_mask.expand(-1, -1, self.n_agents)

        policy_mask = filled_agents * active.squeeze(-1)
        value_mask = policy_mask if getattr(self.args, "use_value_active_masks", True) else filled_agents

        T = rewards.size(1)
        actor_inputs = self._actor_inputs_all(batch, T)
        critic_inputs = self.critic.build_inputs_all(batch, T + 1)
        actor_h_all = batch["actor_hidden"][:, : T + 1]
        reset = th.zeros(bs, T + 1, self.n_agents, device=batch.device)
        reset[:, 1:, :] = episode_end.expand(-1, -1, self.n_agents)

        with th.no_grad():
            old_values, critic_h_all = self._unroll_net(
                critic_inputs, self.critic.init_hidden(bs, device=batch.device), reset, T + 1, critic=True
            )
            old_log_pi, _ = self._actor_logp(actor_inputs[:, :T], actor_h_all[:, :T], actions, avail_actions)

        advantages, returns = self._gae(rewards, old_values, done_mask, policy_mask)
        adv_for_norm = advantages.clone()
        adv_for_norm[policy_mask == 0] = float("nan")
        adv_mean = th.nanmean(adv_for_norm)
        adv_std = th.sqrt(th.nanmean((adv_for_norm - adv_mean) ** 2) + 1e-5)
        advantages = (advantages - adv_mean) / adv_std
        advantages = advantages * policy_mask

        if self.value_norm is not None:
            self.value_norm.update(returns[policy_mask > 0].unsqueeze(-1))
            norm_returns = self.value_norm.normalize(returns.unsqueeze(-1)).squeeze(-1)
        else:
            norm_returns = returns

        chunk = int(getattr(self.args, "data_chunk_length", 10))
        n_mini = int(getattr(self.args, "num_mini_batch", 1))
        n_chunks = max(1, T // chunk)
        usable_t = n_chunks * chunk

        packed = self._pack_sequences(
            {
                "actor_in": actor_inputs[:, :usable_t],
                "critic_in": critic_inputs[:, :usable_t],
                "actor_h": actor_h_all[:, :usable_t],
                "critic_h": critic_h_all[:, :usable_t],
                "actions": actions[:, :usable_t],
                "avail": avail_actions[:, :usable_t],
                "old_log_pi": old_log_pi[:, :usable_t],
                "old_values": old_values[:, :usable_t],
                "adv": advantages[:, :usable_t],
                "ret": norm_returns[:, :usable_t],
                "pmask": policy_mask[:, :usable_t],
                "vmask": value_mask[:, :usable_t],
                "reset": reset[:, :usable_t],
            },
            chunk,
            n_chunks,
        )

        last_pg = last_v = last_ent = last_ratio = 0.0
        last_actor_gn = last_critic_gn = last_pi_max = 0.0
        n_seq = packed["actor_in"].size(0)
        mb_size = max(1, n_seq // max(1, n_mini))

        for _ in range(self.args.epochs):
            perm = th.randperm(n_seq, device=batch.device)
            for start in range(0, n_seq, mb_size):
                idx = perm[start : start + mb_size]
                if idx.numel() == 0:
                    continue
                sample = {k: v[idx] for k, v in packed.items()}
                pg_loss, v_loss, entropy, ratio, actor_gn, critic_gn, pi_max = self._ppo_chunk(sample, chunk)
                last_pg, last_v, last_ent, last_ratio = pg_loss, v_loss, entropy, ratio
                last_actor_gn, last_critic_gn, last_pi_max = actor_gn, critic_gn, pi_max

        extra_stats = self._after_ppo(batch, t_env) or {}

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", last_v, t_env)
            self.logger.log_stat("critic_grad_norm", last_critic_gn, t_env)
            self.logger.log_stat("pg_loss", last_pg, t_env)
            self.logger.log_stat("agent_grad_norm", last_actor_gn, t_env)
            self.logger.log_stat("pi_max", last_pi_max, t_env)
            self.logger.log_stat("advantage_mean", advantages.sum().item() / policy_mask.sum().clamp(min=1).item(), t_env)
            self.logger.log_stat("q_taken_mean", old_values[:, :T].mean().item(), t_env)
            self.logger.log_stat("target_mean", returns.mean().item(), t_env)
            self.logger.log_stat("td_error_abs", (returns - old_values[:, :T]).abs().mean().item(), t_env)
            self.logger.log_stat("entropy", last_ent, t_env)
            self.logger.log_stat("ratio", last_ratio, t_env)
            for key, val in extra_stats.items():
                self.logger.log_stat(key, val, t_env)
            self.log_stats_t = t_env

    def _after_ppo(self, batch: EpisodeBatch, t_env: int):
        return None

    def _actor_inputs_all(self, batch, T):
        bs = batch.batch_size
        outs = []
        for t in range(T):
            x = self.mac._build_inputs(batch, t)
            outs.append(x.view(bs, self.n_agents, -1))
        return th.stack(outs, dim=1)

    def _unroll_net(self, inputs, h0, reset, T, critic=True):
        bs = inputs.size(0)
        h = h0.reshape(bs * self.n_agents, -1)
        values = []
        hiddens = []
        for t in range(T):
            if t > 0:
                h = h * (1.0 - reset[:, t].reshape(bs * self.n_agents, 1))
            hiddens.append(h.view(bs, self.n_agents, -1))
            x = inputs[:, t].reshape(bs * self.n_agents, -1)
            if critic:
                v, h = self.critic.forward_tensor(x, h)
                values.append(v.view(bs, self.n_agents))
            else:
                logits, h = self.mac.agent(x, h)
                values.append(logits.view(bs, self.n_agents, -1))
        stacked_h = th.stack(hiddens, dim=1)
        stacked_v = th.stack(values, dim=1)
        return stacked_v, stacked_h

    def _actor_logp(self, actor_inputs, actor_h, actions, avail_actions):
        bs, T, n_agents, _ = actor_inputs.shape
        logps = []
        ents = []
        for t in range(T):
            x = actor_inputs[:, t].reshape(bs * n_agents, -1)
            h = actor_h[:, t].reshape(bs * n_agents, -1)
            logits, _ = self.mac.agent(x, h)
            logits = logits.view(bs, n_agents, -1)
            logp, ent, _ = self._dist_stats(logits, actions[:, t], avail_actions[:, t])
            logps.append(logp)
            ents.append(ent)
        return th.stack(logps, dim=1), th.stack(ents, dim=1)

    def _dist_stats(self, logits, actions, avail):
        if getattr(self.args, "mask_before_softmax", True):
            logits = logits.clone()
            logits[avail == 0] = -1e10
        pi = F.softmax(logits, dim=-1)
        taken = th.gather(pi, dim=2, index=actions.long()).squeeze(-1)
        logp = th.log(taken + 1e-10)
        ent = -(pi * th.log(pi + 1e-10)).sum(dim=-1)
        pi_max = pi.max(dim=-1)[0]
        return logp, ent, pi_max

    def _pack_sequences(self, tensors, chunk, n_chunks):
        packed = {}
        for key, x in tensors.items():
            # (B, T, A, ...) -> (B*A*n_chunks, chunk, ...)
            x = x.transpose(1, 2).contiguous()
            b, a = x.size(0), x.size(1)
            rest = x.shape[3:]
            x = x[:, :, : n_chunks * chunk]
            if rest:
                x = x.reshape(b * a, n_chunks, chunk, *rest)
                x = x.reshape(b * a * n_chunks, chunk, *rest)
            else:
                x = x.reshape(b * a, n_chunks, chunk)
                x = x.reshape(b * a * n_chunks, chunk)
            packed[key] = x
        return packed

    def _gae(self, rewards, values, done_mask, policy_mask):
        T = rewards.size(1)
        if self.value_norm is not None:
            v = self.value_norm.denormalize(values.unsqueeze(-1)).squeeze(-1)
        else:
            v = values
        advantages = th.zeros_like(rewards)
        gae = th.zeros_like(rewards[:, 0])
        gamma = self.args.gamma
        lam = self.args.gae_lambda
        for t in reversed(range(T)):
            next_v = v[:, t + 1]
            next_mask = done_mask[:, t]
            delta = rewards[:, t] + gamma * next_v * next_mask - v[:, t]
            gae = delta + gamma * lam * next_mask * gae
            advantages[:, t] = gae
        returns = advantages + v[:, :T]
        return advantages.detach(), returns.detach()

    def _ppo_chunk(self, sample, chunk):
        eps = self.args.eps_clip
        huber_delta = float(getattr(self.args, "huber_delta", 10.0))
        entropy_coef = self.args.entropy_coef
        use_huber = getattr(self.args, "use_huber_loss", True)
        use_vclip = getattr(self.args, "use_clipped_value_loss", True)

        n_seq = sample["actor_in"].size(0)
        h_a = sample["actor_h"][:, 0].reshape(n_seq, -1).contiguous().detach()
        h_c = sample["critic_h"][:, 0].reshape(n_seq, -1).contiguous().detach()

        log_pi_list = []
        ent_list = []
        pi_max_list = []
        v_list = []

        for t in range(chunk):
            if t > 0:
                r = sample["reset"][:, t].reshape(n_seq, 1)
                h_a = h_a * (1.0 - r)
                h_c = h_c * (1.0 - r)
            logits, h_a = self.mac.agent(sample["actor_in"][:, t], h_a)
            logits = logits.view(n_seq, 1, -1)
            actions_t = sample["actions"][:, t].view(n_seq, 1, 1)
            avail_t = sample["avail"][:, t]
            if avail_t.dim() == 2:
                avail_t = avail_t.unsqueeze(1)
            logp, ent, pmax = self._dist_stats(logits, actions_t, avail_t)
            log_pi_list.append(logp.squeeze(1))
            ent_list.append(ent.squeeze(1))
            pi_max_list.append(pmax.squeeze(1))
            v, h_c = self.critic.forward_tensor(sample["critic_in"][:, t], h_c)
            v_list.append(v.squeeze(-1))

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

    def _update_targets_hard(self):
        return

    def _update_targets_soft(self, tau):
        return

    def cuda(self):
        self.mac.to(self.args.device)
        self.critic.to(self.args.device)
        if self.value_norm is not None:
            self.value_norm.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        if self.value_norm is not None:
            th.save(self.value_norm.state_dict(), "{}/value_norm.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage)
        )
        if self.value_norm is not None:
            vn_path = "{}/value_norm.th".format(path)
            try:
                self.value_norm.load_state_dict(
                    th.load(vn_path, map_location=lambda storage, loc: storage)
                )
            except FileNotFoundError:
                pass
