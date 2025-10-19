python tools/generate_and_eval.py --ckpt results/ablation/iwslt_encdec/ckpt_epoch10.pth \
  --inputs data/iwslt_test.src.txt --refs data/iwslt_test.ref.txt \
  --out results/ablation/iwslt_encdec/preds.greedy.txt --tokenizer t5-small --device cuda --mode greedy --max_len 80
