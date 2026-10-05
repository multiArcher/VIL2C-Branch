"""Episode return and packet-delay records for a frozen policy."""
import gzip
import json

import numpy as np


def scalar_return(value):
    return float(np.asarray(value, dtype=float).sum())


class ReturnDiagnostics:
    """Episode return and packet-delay records.

    VIL2C does not expose BCRBC latents, so reconstruction and action-agreement
    fields stay absent. Plot scripts that need those fields leave the figure out.
    """

    def __init__(self, directory, batch_size, offset):
        self.directory = directory
        self.offset = offset
        self.rows = [[] for _ in range(batch_size)]
        self.histograms = [{} for _ in range(batch_size)]
        self.clipped = [0 for _ in range(batch_size)]
        self.decisions = gzip.open(directory / "decisions.jsonl.gz", "wt", encoding="utf-8")

    @staticmethod
    def record(runner, active):
        if not active:
            return
        step = runner.t
        data = runner.get_diagnostic_data(active)
        generation_time = runner.batch["obs_gen_t"][:, step, :, 0]
        missing = generation_time < step
        decision_ms = float(getattr(runner.mac, "decision_ms", 0.0))
        diagnostics = runner.diagnostics
        for row_index, env_index in enumerate(active):
            diagnostics.clipped[env_index] += data[row_index]["clipped_count"]
            for delay in data[row_index]["sampled_delays"]:
                key = str(int(delay))
                histograms = diagnostics.histograms[env_index]
                histograms[key] = histograms.get(key, 0) + 1
            for agent in range(runner.mac.n_agents):
                gen = int(generation_time[env_index, agent])
                row = dict(episode=diagnostics.offset + int(env_index), step=int(step), agent=agent,
                           missing=bool(missing[env_index, agent].item()), never_arrived=gen < 0,
                           age=None if gen < 0 else int(step - gen), eligible=False,
                           regime=data[row_index]["regime"], regime_age=data[row_index]["regime_age"],
                           decision_ms=decision_ms)
                diagnostics.decisions.write(json.dumps(row) + "\n")
                diagnostics.rows[env_index].append(row)

    def finish(self, returns, lengths, wins):
        self.decisions.close()
        with (self.directory / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for index, rows in enumerate(self.rows):
                sample_count = len(rows)
                record = dict(episode=self.offset + index, won=bool(wins[index]),
                              episode_return=scalar_return(returns[index]), length=int(lengths[index]),
                              decision_count=0, sample_count=sample_count,
                              missing_count=sum(row["missing"] for row in rows),
                              never_arrived_count=sum(row["never_arrived"] for row in rows),
                              clipped_count=self.clipped[index],
                              delay_histogram=self.histograms[index])
                stream.write(json.dumps(record) + "\n")

    def log(self, logger, t_env):
        pass
