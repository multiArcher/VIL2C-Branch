#!/usr/bin/env bash
set -euo pipefail

# Activate the Python environment and set SC2PATH before running this script.
# Extra Sacred overrides can be appended, e.g. hidden_dim=128 msg_dim=128.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/../.."

exec "${PYTHON:-python}" src/main.py \
  --config=vil2c_ymappo \
  --env-config=sc2 \
  with \
  env_args.map_name=8m_vs_9m \
  epochs=15 \
  num_mini_batch=1 \
  eps_clip=0.05 \
  actor_gain=0.01 \
  use_value_active_masks=False \
  batch_size_run=8 \
  batch_size=8 \
  buffer_size=8 \
  t_max=10000000 \
  seed=1 \
  "$@"
