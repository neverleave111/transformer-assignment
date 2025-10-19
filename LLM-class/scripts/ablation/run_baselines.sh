#!/usr/bin/env bash
set -e
BASE_ARGS="--dataset tiny_shakespeare --epochs 10 --batch_size 64 --seq_len 128 --embed_dim 128 --num_layers 2 --num_heads 4 --ff_dim 512 --log_every 10 --seed 42"

# baseline (full attention, no relative)
./scripts/ablation/run_one.sh baseline ${BASE_ARGS} --attention_type full

# no grad clip
./scripts/ablation/run_one.sh no_grad_clip ${BASE_ARGS} --grad_clip 0

# learned pos
./scripts/ablation/run_one.sh pos_learned ${BASE_ARGS} --learned_pos

# pos removed: we pass a flag --no_pos and train.py will need to respect it.
./scripts/ablation/run_one.sh pos_removed ${BASE_ARGS} --no_pos

# relative pos on
./scripts/ablation/run_one.sh rel_on ${BASE_ARGS} --relative_pos

# linear attention
./scripts/ablation/run_one.sh attn_linear ${BASE_ARGS} --attention_type linear

# local attention window=8
./scripts/ablation/run_one.sh attn_local_w8 ${BASE_ARGS} --attention_type local --local_window 8

# local attention window=16
./scripts/ablation/run_one.sh attn_local_w16 ${BASE_ARGS} --attention_type local --local_window 16
