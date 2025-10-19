#!/usr/bin/env bash
set -e
BASE_ARGS="--dataset tiny_shakespeare --epochs 3 --batch_size 64 --seq_len 128 --log_every 10 --seed 42"

# small model
./scripts/ablation/run_one.sh small_model ${BASE_ARGS} --embed_dim 64 --num_layers 1 --num_heads 1 --ff_dim 128

# base (already run in baseline) - skip or re-run
./scripts/ablation/run_one.sh base_model ${BASE_ARGS} --embed_dim 128 --num_layers 2 --num_heads 4 --ff_dim 512

# big
./scripts/ablation/run_one.sh big_model ${BASE_ARGS} --embed_dim 256 --num_layers 4 --num_heads 8 --ff_dim 1024
