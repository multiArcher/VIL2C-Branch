#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"${PYTHON:-python}" -u src/main.py --config=dama_ddpg --env-config=mpe with \
    name=dama_ddpg_spread env_args.map_name=simple_spread_v3 seed="${SEED:-1}" \
    env_args.time_limit=25 env_args.scenario_args.N=3 \
    common_reward=True reward_scalarisation=sum \
    batch_size_run=4 batch_size=32 buffer_size=2000 dama_history_length=16 \
    t_max=2000000 test_nepisode=32 test_interval=10000 log_interval=10000 \
    runner_log_interval=10000 learner_log_interval=10000 save_model_interval=50000 "$@"
