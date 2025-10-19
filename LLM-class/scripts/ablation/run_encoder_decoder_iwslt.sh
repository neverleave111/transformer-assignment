#!/usr/bin/env bash
set -e
# small sample for speed
BASE="--dataset iwslt2017 --local_data_dir src/data/iwslt2017/en-de --src_lang en --tgt_lang de --seq_len 128 --batch_size 32 --epochs 10 --max_samples 5000 --log_every 20 --seed 42"

# enc-dec baseline
./scripts/ablation/run_one.sh iwslt_encdec ${BASE} --attention_type full

# enc-dec with relative pos
./scripts/ablation/run_one.sh iwslt_encdec_rel ${BASE} --attention_type full --relative_pos

# encoder-only (as LM): treat source==target (not ideal but used for ablation)
./scripts/ablation/run_one.sh iwslt_encoderonly ${BASE} --attention_type full --enc_only
