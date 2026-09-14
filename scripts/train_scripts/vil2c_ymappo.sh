#!/usr/bin/env bash
# Run from the project root with marl_stable activated; inherit SC2PATH.
# Cluster: job.sh should srun this file (or vil2c_ymappo.local.sh).
# VIL2C on YMAPPO, SMAC 5m_vs_6m, t_max=6e6.
# No comm delay (mean/std=0). Progressive reception stays on.
# Env is sc2, not delayed_sc2.

EXPERIMENT_NAME=vil2c_ymappo_5m_vs_6m
MAP_NAME=5m_vs_6m
SEED=1
BATCH_SIZE_RUN=16
BATCH_SIZE=16
BUFFER_SIZE=16
ROLLOUT_LENGTH=400
T_MAX=6000000

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u src/main.py --config=vil2c_ymappo --env-config=sc2 with \
    name="$EXPERIMENT_NAME" env_args.map_name="$MAP_NAME" seed="$SEED" \
    runner=ymappo batch_size_run="$BATCH_SIZE_RUN" \
    batch_size="$BATCH_SIZE" buffer_size="$BUFFER_SIZE" \
    rollout_length="$ROLLOUT_LENGTH" buffer_cpu_only=True \
    use_cuda=True \
    test_nepisode=32 test_interval=50000 log_interval=50000 \
    runner_log_interval=10000 learner_log_interval=10000 \
    save_model_interval=50000 t_max="$T_MAX" \
    comm_gaussian_delay_mean=0.0 comm_gaussian_delay_std=0.0 \
    use_progressive=True "$@"
