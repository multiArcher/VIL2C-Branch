"""Truncated parallel runner for YMAPPO: 8 x rollout_length steps with auto-reset."""

from functools import partial

import numpy as np
import torch as th

from components.episode_buffer import EpisodeBatch
from runners.parallel_runner import ParallelRunner


class YMAPPORunner(ParallelRunner):
    def setup(self, scheme, groups, preprocess, mac):
        self.rollout_length = int(getattr(self.args, "rollout_length", self.episode_limit))
        self.max_seq_length = max(self.episode_limit, self.rollout_length) + 1
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.max_seq_length,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self._train_bootstrapped = False

    def _active_from_avail(self, avail):
        avail = np.asarray(avail)
        if avail.ndim == 1:
            return float(avail[1:].sum() > 0)
        return (avail[..., 1:].sum(axis=-1) > 0).astype(np.float32)

    def _reshape_hidden(self):
        h = self.mac.hidden_states
        if h is not None and h.dim() == 2:
            self.mac.hidden_states = h.view(self.batch_size, self.args.n_agents, -1)

    def _zero_hidden(self, env_idx):
        self._reshape_hidden()
        if self.mac.hidden_states is None:
            return
        self.mac.hidden_states[env_idx] = 0

    def run(self, test_mode=False):
        if test_mode:
            batch = super().run(test_mode=True)
            # Eval uses the same env processes; restart the truncated train stream afterwards.
            self._train_bootstrapped = False
            return batch
        return self._run_truncated()

    def _run_truncated(self):
        self.mac.train()
        if not self._train_bootstrapped:
            self.reset(test_mode=False)
            self.mac.init_hidden(batch_size=self.batch_size)
            self._train_bootstrapped = True
        else:
            # Continue from last observation; new EpisodeBatch each update.
            last_pre = self._last_pre
            self.batch = self.new_batch()
            self.batch.update(last_pre, ts=0)
            self.t = 0
            self.env_steps_this_run = 0

        if self.args.common_reward:
            episode_returns = [0 for _ in range(self.batch_size)]
        else:
            episode_returns = [np.zeros(self.args.n_agents) for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        final_env_infos = []
        completed_returns = []
        completed_lengths = []

        for step in range(self.rollout_length):
            actor_h = self.mac.hidden_states.detach().clone()
            self.batch.update({"actor_hidden": actor_h}, ts=self.t, mark_filled=False)

            avail = self.batch["avail_actions"][:, self.t].cpu().numpy()
            active = self._active_from_avail(avail)
            self.batch.update(
                {"active_masks": th.tensor(active, device=self.batch.device).unsqueeze(-1)},
                ts=self.t,
                mark_filled=False,
            )

            actions = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                bs=list(range(self.batch_size)),
                test_mode=False,
            )
            cpu_actions = actions.to("cpu").numpy()
            self._reshape_hidden()
            self.batch.update({"actions": actions.unsqueeze(1)}, ts=self.t, mark_filled=False)

            for idx, parent_conn in enumerate(self.parent_conns):
                parent_conn.send(("step", cpu_actions[idx]))

            post_transition_data = {"reward": [], "terminated": [], "episode_end": []}
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "obs_delay": [],
                "obs_gen_t": [],
                "obs_fresh_mask": [],
            }

            for idx, parent_conn in enumerate(self.parent_conns):
                data = parent_conn.recv()
                post_transition_data["reward"].append((data["reward"],))
                episode_returns[idx] += data["reward"]
                episode_lengths[idx] += 1
                self.env_steps_this_run += 1

                env_done = bool(data["terminated"])
                time_limit = bool(data["info"].get("episode_limit", False))
                true_term = env_done and not time_limit
                post_transition_data["terminated"].append((true_term,))
                post_transition_data["episode_end"].append((env_done,))

                if env_done:
                    final_env_infos.append(data["info"])
                    completed_returns.append(episode_returns[idx])
                    completed_lengths.append(episode_lengths[idx])
                    if self.args.common_reward:
                        episode_returns[idx] = 0
                    else:
                        episode_returns[idx] = np.zeros(self.args.n_agents)
                    episode_lengths[idx] = 0
                    parent_conn.send(("reset", True))
                    reset_data = parent_conn.recv()
                    pre_transition_data["state"].append(reset_data["state"])
                    pre_transition_data["avail_actions"].append(reset_data["avail_actions"])
                    pre_transition_data["obs"].append(reset_data["obs"])
                    pre_transition_data["obs_delay"].append(reset_data["obs_delay"])
                    pre_transition_data["obs_gen_t"].append(reset_data["obs_gen_t"])
                    pre_transition_data["obs_fresh_mask"].append(reset_data["obs_fresh_mask"])
                    self._zero_hidden(idx)
                else:
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data["obs_delay"].append(data["obs_delay"])
                    pre_transition_data["obs_gen_t"].append(data["obs_gen_t"])
                    pre_transition_data["obs_fresh_mask"].append(data["obs_fresh_mask"])

            self.batch.update(post_transition_data, ts=self.t, mark_filled=False)
            self.t += 1
            self.batch.update(pre_transition_data, ts=self.t, mark_filled=True)

        # Bootstrap observation at t = rollout_length; store hidden/active too.
        actor_h = self.mac.hidden_states.detach().clone()
        self.batch.update({"actor_hidden": actor_h}, ts=self.t, mark_filled=False)
        avail = self.batch["avail_actions"][:, self.t].cpu().numpy()
        active = self._active_from_avail(avail)
        self.batch.update(
            {"active_masks": th.tensor(active, device=self.batch.device).unsqueeze(-1)},
            ts=self.t,
            mark_filled=False,
        )
        self.batch.update(
            {
                "episode_end": [(0,) for _ in range(self.batch_size)],
                "terminated": [(0,) for _ in range(self.batch_size)],
            },
            ts=self.t,
            mark_filled=False,
        )

        self._last_pre = {
            k: [self.batch[k][i, self.t].cpu().numpy() for i in range(self.batch_size)]
            for k in (
                "state",
                "avail_actions",
                "obs",
                "obs_delay",
                "obs_gen_t",
                "obs_fresh_mask",
            )
        }

        self.t_env += self.env_steps_this_run

        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))
        for parent_conn in self.parent_conns:
            parent_conn.recv()

        cur_stats = self.train_stats
        cur_returns = self.train_returns
        infos = [cur_stats] + final_env_infos
        if final_env_infos:
            cur_stats.update(
                {
                    k: sum(d.get(k, 0) for d in infos)
                    for k in set.union(*[set(d) for d in infos])
                }
            )
            cur_stats["n_episodes"] = len(final_env_infos) + cur_stats.get("n_episodes", 0)
            cur_stats["ep_length"] = sum(completed_lengths) + cur_stats.get("ep_length", 0)
            cur_returns.extend(completed_returns)
            if self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
                self._log(cur_returns, cur_stats, "")
                self.log_train_stats_t = self.t_env

        return self.batch
