#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"${PYTHON:-python}" -u src/main.py --config=dama_ddpg --env-config=sc2 with \
    name=dama_ddpg_5m_vs_6m env_args.map_name=5m_vs_6m seed="${SEED:-1}" \
    batch_size_run=4 batch_size=32 buffer_size=500 dama_history_length=16 \
    t_max=6000000 test_nepisode=32 test_interval=50000 log_interval=50000 \
    runner_log_interval=10000 learner_log_interval=10000 save_model_interval=50000 "$@"
