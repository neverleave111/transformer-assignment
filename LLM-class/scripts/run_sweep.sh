#!/usr/bin/env bash
mkdir -p results/sweep
for lr in 1e-4 3e-4 1e-3; do
  for seed in 7 42 123; do
    outdir=results/sweep/lr_${lr}_seed_${seed}
    python src/train.py --dataset tiny_shakespeare --epochs 30 --batch_size 64 --lr ${lr} --seed ${seed} --save_dir ${outdir} --log_every 20
  done
done
