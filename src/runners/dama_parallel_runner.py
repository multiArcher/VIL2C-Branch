"""Keep public rollout behavior; choose tag checkpoints by one team's return."""
import numpy as np

from runners.parallel_runner import ParallelRunner


class DAMAParallelRunner(ParallelRunner):
    def __init__(self, args, logger):
        selected = getattr(args, "dama_checkpoint_agents", None)
        if not args.common_reward and not selected:
            raise ValueError("General-sum DAMA requires dama_checkpoint_agents for model selection")
        super().__init__(args, logger)
        if selected and any(i < 0 or i >= self.env_info["n_agents"] for i in selected):
            self.close_env()
            raise ValueError("dama_checkpoint_agents contains an invalid agent index")

    def _log(self, returns, stats, prefix):
        selected = getattr(self.args, "dama_checkpoint_agents", None)
        selection_score = None
        if not self.args.common_reward and prefix == "test_" and selected:
            selection_score = float(np.asarray(returns)[:, selected].mean())
        super()._log(returns, stats, prefix)
        if selection_score is not None:
            # run.py already prefers this key over total_return_mean. The
            # original individual/total metrics remain available for reporting.
            self.logger.log_stat("metric/test_return_mean", selection_score, self.t_env)
            self.logger.log_stat("dama/checkpoint_team_return_mean", selection_score, self.t_env)
