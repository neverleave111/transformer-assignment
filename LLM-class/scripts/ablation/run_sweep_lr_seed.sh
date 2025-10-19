#!/usr/bin/env bash
set -e
BASE="--dataset tiny_shakespeare --epochs 3 --batch_size 64 --seq_len 128 --log_every 10"
for lr in 1e-4 3e-4 1e-3; do
  for seed in 7 42 123; do
    RUN=lr_${lr}_seed_${seed}
    ./scripts/ablation/run_one.sh ${RUN} ${BASE} --lr ${lr} --seed ${seed}
  done
done
