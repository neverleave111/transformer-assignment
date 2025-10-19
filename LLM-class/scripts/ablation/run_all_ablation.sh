#!/usr/bin/env bash
set -e
mkdir -p results/ablation
bash scripts/ablation/run_baselines.sh
bash scripts/ablation/run_scale_heads_layers.sh
bash scripts/ablation/run_sweep_lr_seed.sh
# run iwslt tests if data present
bash scripts/ablation/run_encoder_decoder_iwslt.sh || echo "iwslt runs skipped (missing data or error)"
echo "All requested ablation runs launched. Check results/ablation/*"
