#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Train both teams with separate rewards; never sum opponent rewards for learning.
"${PYTHON:-python}" -u src/main.py --config=dama_ddpg --env-config=mpe with \
    name=dama_ddpg_tag env_args.map_name=simple_tag_v3 seed="${SEED:-1}" \
    env_args.time_limit=25 env_args.scenario_args.num_adversaries=3 \
    env_args.scenario_args.num_good=1 env_args.scenario_args.num_obstacles=2 \
    common_reward=False 'dama_checkpoint_agents=[0,1,2]' \
    batch_size_run=4 batch_size=32 buffer_size=2000 dama_history_length=16 \
    t_max=2000000 test_nepisode=32 test_interval=10000 log_interval=10000 \
    runner_log_interval=10000 learner_log_interval=10000 save_model_interval=50000 "$@"
