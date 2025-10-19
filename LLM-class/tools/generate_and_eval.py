#!/usr/bin/env python3
# tools/generate_and_eval.py
"""
加载 checkpoint，批量生成（greedy 或 beam），计算 sacreBLEU 与 ROUGE-L。
示例：
python tools/generate_and_eval.py --ckpt results/ablation/iwslt_encdec/ckpt_epoch3.pth \
  --inputs data/iwslt_test.src.txt --refs data/iwslt_test.ref.txt \
  --out results/ablation/iwslt_encdec/preds.greedy.txt --tokenizer t5-small --device cuda --mode greedy --max_len 80
"""
import os, json, argparse, math, sys
import torch
from tqdm import tqdm
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import sys
import os
# 添加src所在的父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import model classes (try both import styles)
try:
    from src.model import Encoder, Decoder
except Exception:
    from model import Encoder, Decoder

def load_tokenizer(name):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name, use_fast=True)
    except Exception as e:
        raise RuntimeError("install transformers and provide tokenizer name") from e

def build_models_from_ckpt(ckpt_path, tokenizer, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = ckpt.get("args", {}) or {}
    # infer vocab_size from tokenizer
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else tokenizer.vocab_size()
    # build encoder & decoder using saved args if present; otherwise use defaults
    enc_args = dict(
        vocab_size=vocab_size,
        embed_dim=args.get("embed_dim", 128),
        num_layers=args.get("num_layers", 2),
        num_heads=args.get("num_heads", 4),
        ff_dim=args.get("ff_dim", 512),
        max_len=args.get("seq_len", 128),
        dropout=args.get("dropout", 0.1),
        learned_pos=args.get("learned_pos", False),
        attention_type=args.get("attention_type", "full"),
        relative_pos=args.get("relative_pos", False),
        max_rel=args.get("max_relative_position", 16),
        local_window=args.get("local_window", 8),
    )
    dec_args = dict(
        vocab_size=vocab_size,
        embed_dim=args.get("embed_dim", 128),
        num_layers=args.get("num_decoder_layers", 2),
        num_heads=args.get("num_heads", 4),
        ff_dim=args.get("ff_dim", 512),
        max_len=args.get("seq_len", 128),
        dropout=args.get("dropout", 0.1),
        learned_pos=args.get("learned_pos", False),
        attention_type=args.get("attention_type", "full"),
        relative_pos=args.get("relative_pos", False),
        max_rel=args.get("max_relative_position", 16),
        local_window=args.get("local_window", 8),
    )
    encoder = Encoder(**enc_args)
    decoder = Decoder(**dec_args)
    # load states (keys might differ depending on how checkpoint saved)
    if "encoder_state" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state"])
    elif "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
    else:
        print("Warning: encoder weights not found in ckpt keys:", list(ckpt.keys()))
    if "decoder_state" in ckpt:
        decoder.load_state_dict(ckpt["decoder_state"])
    elif "decoder_state_dict" in ckpt:
        decoder.load_state_dict(ckpt["decoder_state_dict"])
    else:
        print("Warning: decoder weights not found in ckpt keys:", list(ckpt.keys()))
    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder, args

def greedy_decode_batch(encoder, decoder, tokenizer, inputs, device, input_max_len=128, max_len=80):
    preds = []
    for src in tqdm(inputs, desc="Decoding"):
        # ensure input is truncated/padded to model max length to avoid pos overflow
        enc_in = tokenizer(src, return_tensors="pt", truncation=True, max_length=input_max_len, padding="max_length")
        input_ids = enc_in["input_ids"].to(device)
        # run encoder
        with torch.no_grad():
            enc_out, _ = encoder(input_ids)
        # greedy decode stepwise (batch size 1)
        start_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        dec_input = torch.tensor([[start_id]], device=device)
        out_ids = []
        for _ in range(max_len):
            with torch.no_grad():
                logits, _, _ = decoder(dec_input, enc_out)
            next_logits = logits[:, -1, :]  # (1, V)
            next_id = torch.argmax(next_logits, dim=-1).unsqueeze(1)  # (1,1)
            nid = next_id.item()
            if nid == tokenizer.eos_token_id or nid == tokenizer.pad_token_id:
                break
            out_ids.append(nid)
            dec_input = torch.cat([dec_input, next_id], dim=1)
        pred = tokenizer.decode(out_ids, skip_special_tokens=True)
        preds.append(pred)
    return preds

def beam_decode_single(encoder, decoder, tokenizer, src, device, input_max_len=128, max_len=80, beam_size=4):
    enc_in = tokenizer(src, return_tensors="pt", truncation=True, max_length=input_max_len, padding="max_length")
    input_ids = enc_in["input_ids"].to(device)
    with torch.no_grad():
        enc_out, _ = encoder(input_ids)
    # simple beam search (not optimized)
    start_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    beams = [(0.0, torch.tensor([[start_id]], device=device))]
    completed = []
    for _ in range(max_len):
        new_beams = []
        for score, dec_input in beams:
            with torch.no_grad():
                logits, _, _ = decoder(dec_input, enc_out)
            next_logits = logits[:, -1, :]  # (1, V)
            logp = torch.nn.functional.log_softmax(next_logits, dim=-1).squeeze(0)  # (V,)
            topk = torch.topk(logp, k=min(beam_size, logp.size(0)))
            for kidx in range(topk.values.size(0)):
                nid = int(topk.indices[kidx].item())
                nscore = score + float(topk.values[kidx].item())
                new_input = torch.cat([dec_input, torch.tensor([[nid]], device=device)], dim=1)
                if nid == tokenizer.eos_token_id:
                    completed.append((nscore, new_input))
                else:
                    new_beams.append((nscore, new_input))
        new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        beams = new_beams
        if len(beams) == 0:
            break
    if len(completed) > 0:
        best = max(completed, key=lambda x: x[0])
        seq = best[1].squeeze(0).tolist()
    elif len(beams) > 0:
        seq = beams[0][1].squeeze(0).tolist()
    else:
        seq = []
    if len(seq) > 0 and seq[0] == tokenizer.pad_token_id:
        seq = seq[1:]
    seq = [i for i in seq if i != tokenizer.eos_token_id and i != tokenizer.pad_token_id]
    pred = tokenizer.decode(seq, skip_special_tokens=True)
    return pred

def batch_beam_decode(encoder, decoder, tokenizer, inputs, device, input_max_len=128, max_len=80, beam_size=4):
    preds = []
    for src in tqdm(inputs, desc="Beam decoding"):
        preds.append(beam_decode_single(encoder, decoder, tokenizer, src, device, input_max_len=input_max_len, max_len=max_len, beam_size=beam_size))
    return preds

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to checkpoint (encoder+decoder)")
    p.add_argument("--inputs", required=True, help="input file (one source per line)")
    p.add_argument("--refs", required=True, help="reference file (one ref per line)")
    p.add_argument("--out", required=True, help="where to write preds (one per line)")
    p.add_argument("--tokenizer", default="t5-small")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_len", type=int, default=80, help="max generation length")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--mode", choices=["greedy","beam"], default="greedy")
    p.add_argument("--beam_size", type=int, default=4)
    args = p.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    encoder, decoder, ckpt_args = build_models_from_ckpt(args.ckpt, tokenizer, args.device)

    # derive input_max_len from checkpoint args if available (fallback to args.max_len or 128)
    input_max_len = ckpt_args.get("seq_len") if ckpt_args and isinstance(ckpt_args.get("seq_len"), int) else None
    if input_max_len is None:
        # if ckpt didn't include seq_len, fallback to args.max_len or 128
        input_max_len = max(128, args.max_len)
    # but we do not want generation max_len < input truncation length; ensure reasonable bound
    input_max_len = int(input_max_len)

    inputs = read_lines(args.inputs)
    refs = read_lines(args.refs)
    assert len(inputs) == len(refs), "inputs/refs length mismatch"

    if args.mode == "greedy":
        preds = greedy_decode_batch(encoder, decoder, tokenizer, inputs, args.device, input_max_len=input_max_len, max_len=args.max_len)
    else:
        preds = batch_beam_decode(encoder, decoder, tokenizer, inputs, args.device, input_max_len=input_max_len, max_len=args.max_len, beam_size=args.beam_size)

    # write preds
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p.replace("\n", " ") + "\n")

    # compute BLEU and ROUGE-L
    bleu = corpus_bleu(preds, [refs])
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = sum([scorer.score(r,p)['rougeL'].fmeasure for r,p in zip(refs,preds)]) / len(preds)

    metrics = {"BLEU": bleu.score, "BLEU_detail": bleu.format(), "ROUGE_L": rouge_l, "num_samples": len(preds)}
    print("Metrics:", metrics)
    # save metrics
    metrics_path = os.path.splitext(args.out)[0] + ".metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("Saved preds to", args.out, "and metrics to", metrics_path)

if __name__ == "__main__":
    main()
