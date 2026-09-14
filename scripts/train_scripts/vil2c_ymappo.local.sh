#!/usr/bin/env bash
# Cluster alias for job.sh: srun bash scripts/train_scripts/vil2c_ymappo.local.sh
exec "$(cd "$(dirname "$0")" && pwd)/vil2c_ymappo.sh" "$@"
