#!/usr/bin/env bash
set -euo pipefail

SIZES=("1MB" "10MB" "100MB" "1GB")
GPUS=(2 4 8)

for sz in "${SIZES[@]}"; do
  for g in "${GPUS[@]}"; do
    echo "=========================================="
    echo "Running with ${g} GPUs, tensor_size=${sz}"
    echo "=========================================="
    srun --nodes=1 --gpus-per-node=${g} uv run dist_comm_single_node.py --tensor_size=${sz}
    echo
  done
done
