"""Discrete DAMA/MADDPG update with executed-history state augmentation.

This adaptation deliberately trains only on undelayed observations. Terminal
rewards remain in the loss; terminal flags suppress bootstrapping, not training.
"""
import copy
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim import Adam

from learners.learner import Learner
from utils.maker import CriticMaker


class DAMALearner(Learner):
    def __init__(self, mac, scheme, logger, args):
        self.args, self.logger, self.mac = args, logger, mac
        if args.standardise_rewards or args.standardise_returns:
            raise ValueError("DAMA currently uses raw rewards/returns; disable standardisation")
        if not 0 < args.target_update_interval_or_tau <= 1:
            raise ValueError("DAMA requires a soft-update tau in (0, 1]")
        self.target_mac = copy.deepcopy(mac)
        self.critic = CriticMaker.make(args.critic_type, scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_mac.requires_grad_(False)
        self.target_critic.requires_grad_(False)
        self.agent_optimiser = Adam(mac.parameters(), lr=args.lr)
        self.critic_optimiser = Adam(self.critic.parameters(), lr=args.critic_lr)
        self.log_stats_t = -args.learner_log_interval - 1
        self.training_steps = 0

    @staticmethod
    def transition_mask(batch):
        # A filled final observation is not itself a transition. Requiring the
        # next observation also excludes a shorter episode's timeout placeholder.
        mask = batch["filled"][:, :-1].float() * batch["filled"][:, 1:].float()
        # Keep the terminal transition itself, exclude subsequent padding.
        if mask.shape[1] > 1:
            mask[:, 1:] *= (1 - batch["terminated"][:, :-2].float()).cumprod(dim=1)
        return mask

    def train(self, batch, t_env, episode_num):
        mask = self.transition_mask(batch)
        if mask.sum() == 0:
            return
        if self.args.dama_require_no_delay_train:
            times = torch.arange(batch.max_seq_length, device=batch.device).view(1, -1, 1, 1)
            present = batch["filled"].bool().unsqueeze(2)
            if ((batch["obs_gen_t"] != times) & present).any():
                raise ValueError("DAMA training must use current observations, not delayed batches")
        inputs = self.mac.inputs.sequence(batch)
        rewards = batch["reward"][:, :-1]
        if self.args.common_reward:
            rewards = rewards.expand(-1, -1, self.args.n_agents)
        rewards = rewards.unsqueeze(-1)
        term = batch["terminated"][:, :-1].float().unsqueeze(2)
        weights = mask.unsqueeze(2).expand(-1, -1, self.args.n_agents, -1)
        denominator = weights.sum().clamp_min(1)
        with torch.no_grad():
            logits = self.target_mac.logits(inputs[:, 1:], batch["avail_actions"][:, 1:])
            next_actions = F.one_hot(logits.argmax(-1), self.args.n_actions).float()
            targets = rewards + self.args.gamma * (1 - term) * self.target_critic(
                inputs[:, 1:], next_actions)
        values = self.critic(inputs[:, :-1], batch["actions_onehot"][:, :-1])
        error = values - targets
        critic_loss = (error.square() * weights).sum() / denominator
        self.critic_optimiser.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_norm_clip)
        self.critic_optimiser.step()

        logits = self.mac.logits(inputs[:, :-1], batch["avail_actions"][:, :-1])
        chosen = F.gumbel_softmax(logits, tau=self.args.dama_gumbel_temperature, hard=True)
        # Each actor optimizes ONLY its own critic. Other agents' actions are
        # detached, so competitive rewards cannot update the opponent's actor.
        self.critic.requires_grad_(False)
        try:
            actor_values = []
            for i in range(self.args.n_agents):
                joint = torch.stack([chosen[..., j, :] if i == j else chosen[..., j, :].detach()
                                     for j in range(self.args.n_agents)], dim=-2)
                actor_values.append(self.critic.agent_value(inputs[:, :-1], joint, i))
            actor_values = torch.stack(actor_values, dim=-2)
            valid_logits = batch["avail_actions"][:, :-1].bool()
            raw_logits = torch.where(valid_logits, logits, 0)
            reg_weights = mask.unsqueeze(2) * valid_logits
            regularizer = (raw_logits.square() * reg_weights).sum() / reg_weights.sum().clamp_min(1)
            actor_loss = -(actor_values * weights).sum() / denominator + self.args.reg * regularizer
            self.agent_optimiser.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(self.mac.parameters(), self.args.grad_norm_clip)
            self.agent_optimiser.step()
        finally:
            self.critic.requires_grad_(True)
        self._update_targets_soft(self.args.target_update_interval_or_tau)
        self.training_steps += 1
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            stats = {"critic_loss": critic_loss.item(), "pg_loss": actor_loss.item(),
                     "critic_grad_norm": float(critic_grad), "agent_grad_norm": float(actor_grad),
                     "td_error_abs": (error.abs() * weights).sum().item() / denominator.item()}
            for name, value in stats.items():
                self.logger.log_stat(name, value, t_env)
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        with torch.no_grad():
            for target, source in ((self.target_mac, self.mac), (self.target_critic, self.critic)):
                for target_param, param in zip(target.parameters(), source.parameters()):
                    target_param.lerp_(param, tau)

    def to(self, device):
        for module in (self.mac, self.target_mac, self.critic, self.target_critic):
            module.to(device)
        for optimizer in (self.agent_optimiser, self.critic_optimiser):
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        return self

    def cuda(self):
        return self.to(self.args.device)

    def save_models(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.mac.save_models(path)
        torch.save(self.critic.state_dict(), path / "critic.th")
        torch.save(dict(target_mac=self.target_mac.state_dict(),
                        target_critic=self.target_critic.state_dict(),
                        agent_opt=self.agent_optimiser.state_dict(),
                        critic_opt=self.critic_optimiser.state_dict(),
                        training_steps=self.training_steps), path / "training_state.th")

    def load_models(self, path):
        path = Path(path)
        device = next(self.mac.parameters()).device
        self.mac.load_models(path)
        self.critic.load_state_dict(torch.load(path / "critic.th", map_location=device, weights_only=True))
        state = torch.load(path / "training_state.th", map_location=device, weights_only=True)
        self.target_mac.load_state_dict(state["target_mac"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.agent_optimiser.load_state_dict(state["agent_opt"])
        self.critic_optimiser.load_state_dict(state["critic_opt"])
        self.training_steps = state["training_steps"]
