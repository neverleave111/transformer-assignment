# python src/evaluate.py --task lm \
#   --dataset tiny_shakespeare \
#   --checkpoint results/tiny_shakespeare/ckpt_epoch30.pth \
#   --batch_size 64 --seq_len 128 \
#   --embed_dim 128 --num_layers 2 --num_heads 4 --ff_dim 512 \
#   --output_file results/tiny_eval.json

# python src/evaluate.py --task seq2seq \
#   --dataset iwslt2017 \
#   --local_data_dir src/data/iwslt2017/en-de \
#   --checkpoint results/iwslt_en_de/ckpt_epoch30.pth \
#   --batch_size 16 --seq_len 128 \
#   --tokenizer_name t5-small \
#   --embed_dim 128 --num_layers 2 --num_decoder_layers 2 --num_heads 4 --ff_dim 512 \
#   --output_file results/iwslt_eval.json

python src/evaluate.py --task seq2seq \
  --dataset cnn_dailymail --dataset_config 3.0.0 \
  --checkpoint results/cnn_small/ckpt_epoch30.pth \
  --batch_size 8 --seq_len 256 --tokenizer_name t5-small \
  --max_samples 2000 --output_file results/cnn_eval.json
