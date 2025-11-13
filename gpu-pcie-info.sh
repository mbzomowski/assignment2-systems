#!/usr/bin/env bash

for id in {0..7}; do
  echo "GPU $id:"
  nvidia-smi --id=$id \
    --query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max \
    --format=csv
  echo
done
