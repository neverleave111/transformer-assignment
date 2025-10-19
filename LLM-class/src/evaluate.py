#!/usr/bin/env python3
# src/evaluate.py
"""
Evaluation script for LM and Seq2Seq models.

Usage examples:
# LM evaluation (tiny_shakespeare):
python src/evaluate.py --task lm --dataset tiny_shakespeare --checkpoint results/tiny_test/ckpt_epoch3.pth --batch_size 64 --seq_len 128 --max_samples 10000

# Seq2Seq evaluation (IWSLT local):
python src/evaluate.py --task seq2seq --dataset iwslt2017 --local_data_dir src/data/iwslt2017/en-de \
    --checkpoint results/iwslt_local/ckpt_epoch5.pth --batch_size 16 --seq_len 128 --max_samples 2000 --num_beams 1

# Seq2Seq evaluation with sacrebleu (if installed)
pip install sacrebleu
python src/evaluate.py --task seq2seq --dataset iwslt2017 --local_data_dir src/data/iwslt2017/en-de \
    --checkpoint results/iwslt_local/ckpt_epoch5.pth --batch_size 16 --seq_len 128 --max_samples 2000 --num_beams 4
"""
import os
import sys
import argparse
import json
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# try importing project modules robustly
try:
    from data import build_seq2seq_dataloader, build_char_dataset_from_texts
    from model import Encoder, Decoder
    from utils import count_parameters
except Exception:
    # if script executed from project root without PYTHONPATH
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.data import build_seq2seq_dataloader, build_char_dataset_from_texts
    from src.model import Encoder, Decoder
    from src.utils import count_parameters

# optional evaluation libs
_have_sacrebleu = True
try:
    import sacrebleu
except Exception:
    _have_sacrebleu = False

_have_nltk = True
try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
except Exception:
    _have_nltk = False

# simple fallback BLEU (very crude), returns BLEU-4 corpus score
def simple_corpus_bleu(references, hypotheses):
    # references: list of list of ref strings (we'll token split on whitespace)
    # hypotheses: list of hyp strings
    import math, collections
    def ngram_counts(tokens, n):
        c = collections.Counter()
        for i in range(len(tokens)-n+1):
            c[tuple(tokens[i:i+n])] += 1
        return c
    def clipped_precision(refs, hyp, n):
        hyp_tokens = hyp.split()
        hyp_ngrams = ngram_counts(hyp_tokens, n)
        max_counts = {}
        for ref in refs:
            ref_tokens = ref.split()
            ref_counts = ngram_counts(ref_tokens, n)
            for k,v in ref_counts.items():
                max_counts[k] = max(max_counts.get(k,0), v)
        clipped = 0
        total = 0
        for k,v in hyp_ngrams.items():
            clipped += min(v, max_counts.get(k,0))
            total += v
        return (clipped, total)
    precisions = []
    for n in range(1,5):
        num = 0; den = 0
        for refs, hyp in zip(references, hypotheses):
            c, t = clipped_precision(refs, hyp, n)
            num += c; den += t
        p = (num / den) if den>0 else 0.0
        precisions.append(p if p>0 else 1e-9)
    # brevity penalty
    hyp_len = sum(len(h.split()) for h in hypotheses)
    ref_len = 0
    for refs in references:
        # choose ref with closest length to hyp (corpus-level: use sum of best match)
        # easier: use average ref length
        ref_len += sum(len(r.split()) for r in refs)/len(refs)
    bp = 1.0
    if hyp_len == 0:
        bp = 0.0
    else:
        if hyp_len < ref_len:
            bp = math.exp(1 - ref_len / hyp_len)
    log_p = sum((1/4)*math.log(p) for p in precisions)
    bleu = bp * math.exp(log_p)
    return 100.0 * bleu

def load_checkpoint(checkpoint_path, map_location='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    return ckpt

def evaluate_lm(args, device):
    # Build data loader and tokenizer (char-level)
    print("Building char dataset...")
    # Need texts to build tokenizer; build_char_dataset_from_texts expects list of texts.
    # Use build_char_dataset_from_texts via data module; it returns (loader, tokenizer)
    # For evaluation we want dataloader (X,Y)
    # The user may have used datasets when training; to be consistent, we just build from same raw if possible:
    # If args.local_data_dir provided and contains a 'tiny_shakespeare.txt', try to load; else fallback to default behavior in function by passing sample texts.
    raw_texts = None
    if args.dataset == 'tiny_shakespeare':
        # try to load HF dataset via datasets if available? We'll rely on data.py behaviour: build_char_dataset_from_texts expects texts.
        # If there exists dataset file path, try to read
        candidate = os.path.join(args.local_data_dir or '', 'tiny_shakespeare.txt')
        if args.local_data_dir and os.path.exists(candidate):
            with open(candidate, 'r', encoding='utf-8', errors='ignore') as f:
                raw_texts = [f.read()]
    if raw_texts is None:
        # fallback: small builtin sample (same as training fallback)
        raw_texts = [
            "First Citizen: Before we proceed any further, hear me speak.",
            "All: Speak, speak.",
            "First Citizen: You are all resolved rather to die than to famish?",
            "All: Resolved. resolved.",
            "KING: You shall do well."
        ] * 200

    loader, tokenizer = build_char_dataset_from_texts(raw_texts, seq_len=args.seq_len, batch_size=args.batch_size, stride=1, shuffle=False)
    print(f"Eval loader len: {len(loader.dataset)} batches (dataset size {len(loader.dataset)})")
    # build model (Encoder + lm_head)
    vocab_size = tokenizer.vocab_size()
    encoder = Encoder(vocab_size=vocab_size,
                      embed_dim=args.embed_dim,
                      num_layers=args.num_layers,
                      num_heads=args.num_heads,
                      ff_dim=args.ff_dim,
                      max_len=args.seq_len,
                      dropout=args.dropout,
                      learned_pos=args.learned_pos,
                      attention_type=args.attention_type,
                      relative_pos=args.relative_pos,
                      max_rel=args.max_relative_position,
                      local_window=args.local_window)
    lm_head = torch.nn.Linear(args.embed_dim, vocab_size)
    encoder.to(device); lm_head.to(device)
    # load checkpoint
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    # try multiple keys
    if 'encoder_state_dict' in ckpt:
        encoder.load_state_dict(ckpt['encoder_state_dict'])
    elif 'encoder_state' in ckpt:
        encoder.load_state_dict(ckpt['encoder_state'])
    else:
        print("Warning: encoder state not found in checkpoint keys:", list(ckpt.keys()))
    if 'lm_head_state_dict' in ckpt:
        lm_head.load_state_dict(ckpt['lm_head_state_dict'])
    elif 'lm_head' in ckpt:
        try:
            lm_head.load_state_dict(ckpt['lm_head'])
        except Exception:
            print("lm_head key present but failed to load.")
    else:
        print("Warning: lm_head state not found in checkpoint.")
    encoder.eval(); lm_head.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="LM eval"):
            xb = xb.to(device)
            yb = yb.to(device)
            enc_out, _ = encoder(xb)  # expect (B,T,embed)
            logits = lm_head(enc_out)  # (B,T,V)
            B, T, V = logits.size()
            loss = criterion(logits.view(B*T, V), yb.view(B*T))
            total_loss += loss.item() * (B*T)
            total_tokens += (B*T)
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    print(f"LM Eval — avg_loss: {avg_loss:.6f}, ppl: {ppl:.3f}")
    out = {'avg_loss': avg_loss, 'ppl': ppl, 'total_tokens': total_tokens}
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
    return out

def greedy_decode_seq2seq(encoder, decoder, tokenizer, enc_inputs, max_len, device, bos_token_id=None, eos_token_id=None):
    """
    Simple greedy decoding for decoder implementation in this repo.
    enc_inputs: tensor (B, T_enc) input ids
    returns: list of generated token id lists (per example)
    """
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        enc_out, _ = encoder(enc_inputs.to(device))
        B = enc_inputs.size(0)
        # start tokens: use pad token idx 0 or tokenizer.pad_token_id if available
        if bos_token_id is None:
            try:
                bos = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
            except Exception:
                bos = 0
        else:
            bos = bos_token_id
        ys = torch.full((B, 1), bos, dtype=torch.long, device=device)
        finished = [False]*B
        results = [[] for _ in range(B)]
        for step in range(max_len):
            # decoder forward expects dec_input and encoder output
            logits, _, _ = decoder(ys, enc_out)  # logits (B, T, V)
            next_logits = logits[:, -1, :]  # (B, V)
            next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B,1)
            ys = torch.cat([ys, next_tokens], dim=1)
            for i in range(B):
                tok = next_tokens[i].item()
                if eos_token_id is not None and tok == eos_token_id:
                    finished[i] = True
                results[i].append(tok)
            if all(finished):
                break
    return results

def evaluate_seq2seq(args, device):
    # build dataloader and tokenizer
    print("Building seq2seq dataloader...")
    loader, tokenizer = build_seq2seq_dataloader(args.dataset,
                                                tokenizer_name=args.tokenizer_name,
                                                seq_len=args.seq_len,
                                                batch_size=args.batch_size,
                                                split='test' if args.eval_split=='test' else 'train',
                                                src_lang=args.src_lang,
                                                tgt_lang=args.tgt_lang,
                                                dataset_config=args.dataset_config,
                                                max_samples=args.max_samples,
                                                shuffle=False,
                                                dataset_local_dir=args.local_data_dir)
    # build encoder & decoder
    # obtain vocab size from tokenizer
    try:
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else tokenizer.vocab_size()
    except Exception:
        # fallback
        vocab_size = getattr(tokenizer, 'vocab_size', None)
        if vocab_size is None:
            raise RuntimeError("Cannot determine tokenizer vocab size.")
    encoder = Encoder(vocab_size=vocab_size,
                      embed_dim=args.embed_dim,
                      num_layers=args.num_layers,
                      num_heads=args.num_heads,
                      ff_dim=args.ff_dim,
                      max_len=args.seq_len,
                      dropout=args.dropout,
                      learned_pos=args.learned_pos,
                      attention_type=args.attention_type,
                      relative_pos=args.relative_pos,
                      max_rel=args.max_relative_position,
                      local_window=args.local_window)
    decoder = Decoder(vocab_size=vocab_size,
                      embed_dim=args.embed_dim,
                      num_layers=args.num_decoder_layers,
                      num_heads=args.num_heads,
                      ff_dim=args.ff_dim,
                      max_len=args.seq_len,
                      dropout=args.dropout,
                      learned_pos=args.learned_pos,
                      attention_type=args.attention_type,
                      relative_pos=args.relative_pos,
                      max_rel=args.max_relative_position,
                      local_window=args.local_window)
    encoder.to(device); decoder.to(device)
    # load ckpt
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    if 'encoder_state_dict' in ckpt:
        encoder.load_state_dict(ckpt['encoder_state_dict'])
    elif 'encoder_state' in ckpt:
        encoder.load_state_dict(ckpt['encoder_state'])
    else:
        print("Warning: encoder state not found in checkpoint keys:", list(ckpt.keys()))
    if 'decoder_state_dict' in ckpt:
        decoder.load_state_dict(ckpt['decoder_state_dict'])
    elif 'decoder_state' in ckpt:
        decoder.load_state_dict(ckpt['decoder_state'])
    else:
        print("Warning: decoder state not found in checkpoint.")
    # prepare evaluation lists
    hyps = []
    refs = []
    total_loss = 0.0
    tot_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    # iterate
    for batch in tqdm(loader, desc="Seq2Seq eval"):
        # batch: (input_ids, attention_mask, label_ids)
        if len(batch) == 3:
            input_ids, attention_mask, label_ids = batch
        else:
            # support datasets that return dicts
            input_ids = batch[0]; attention_mask = batch[1]; label_ids = batch[2]
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)
        # compute loss
        decoder.train()  # use decoder forward (it returns logits, attn, etc.)
        with torch.no_grad():
            # prepare decoder input: replace -100 with pad token id 0
            dec_in = label_ids.clone()
            dec_in[dec_in == -100] = 0
            logits, _, _ = decoder(dec_in, encoder(input_ids)[0])
            B, T, V = logits.size()
            loss = criterion(logits.view(B*T, V), label_ids.view(B*T))
            total_loss += loss.item()
            tot_tokens += (label_ids != -100).sum().item()
        # decoding (greedy or beams)
        if args.num_beams <= 1:
            # greedy decoding using simple loop
            gen_ids = greedy_decode_seq2seq(encoder, decoder, tokenizer, input_ids, max_len=args.seq_len, device=device,
                                            bos_token_id=getattr(tokenizer, 'pad_token_id', 0), eos_token_id=getattr(tokenizer, 'eos_token_id', None))
            # convert ids to strings
            for i in range(len(gen_ids)):
                # decode generator returns list of token ids (possibly includes pad/eos)
                toks = [t for t in gen_ids[i] if t is not None]
                try:
                    text = tokenizer.decode(toks, skip_special_tokens=True) if hasattr(tokenizer, 'decode') else " ".join(map(str,toks))
                except Exception:
                    # tokenizer may be huggingface one expecting list or numpy
                    try:
                        text = tokenizer.decode(toks, skip_special_tokens=True)
                    except Exception:
                        text = " ".join(map(str,toks))
                hyps.append(text)
            # refs: get label strings from tokenizer
            # labels may have -100; replace -100 with pad id and decode
            lab = label_ids.cpu().numpy()
            for i in range(lab.shape[0]):
                labs = lab[i].tolist()
                labs = [x if x!=-100 else getattr(tokenizer, 'pad_token_id', 0) for x in labs]
                try:
                    r = tokenizer.decode(labs, skip_special_tokens=True)
                except Exception:
                    r = " ".join(str(x) for x in labs)
                refs.append([r])  # sacrebleu expects list of references per example
        else:
            # beam search not implemented fully; fallback to greedy for now and warn
            print("Beam decoding not implemented in this script; falling back to greedy.")
            gen_ids = greedy_decode_seq2seq(encoder, decoder, tokenizer, input_ids, max_len=args.seq_len, device=device,
                                            bos_token_id=getattr(tokenizer, 'pad_token_id', 0), eos_token_id=getattr(tokenizer, 'eos_token_id', None))
            for i in range(len(gen_ids)):
                toks = [t for t in gen_ids[i] if t is not None]
                try:
                    text = tokenizer.decode(toks, skip_special_tokens=True)
                except Exception:
                    text = " ".join(map(str,toks))
                hyps.append(text)
            lab = label_ids.cpu().numpy()
            for i in range(lab.shape[0]):
                labs = lab[i].tolist()
                labs = [x if x!=-100 else getattr(tokenizer, 'pad_token_id', 0) for x in labs]
                try:
                    r = tokenizer.decode(labs, skip_special_tokens=True)
                except Exception:
                    r = " ".join(str(x) for x in labs)
                refs.append([r])
    # compute ppl and loss
    avg_loss = total_loss / max(1, tot_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    # compute BLEU
    # prepare references and hyps as required by sacrebleu: refs list-of-lists -> [ref1_list, ref2_list,...] or list of reference strings?
    # sacrebleu.corpus_bleu expects list of hypothesis strings and list of list of reference strings (refs per reference)
    if _have_sacrebleu:
        # sacrebleu wants refs as list of reference corpora: [[ref1, ref2, ...], [ref1b,...]]
        # convert our refs (list of [r]) to list of lists
        flat_refs = [r[0] for r in refs]
        bleu = sacrebleu.corpus_bleu(hyps, [flat_refs])
        bleu_score = bleu.score
    elif _have_nltk:
        # nltk expects references as list of list of token lists
        tokenized_refs = [[r[0].split()] for r in refs]
        tokenized_hyps = [h.split() for h in hyps]
        smoothie = SmoothingFunction().method4
        bleu_score = corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smoothie)*100.0
    else:
        bleu_score = simple_corpus_bleu(refs, hyps)
    print(f"Seq2Seq Eval — avg_token_loss: {avg_loss:.6f}, ppl: {ppl:.3f}, BLEU: {bleu_score:.3f}")
    out = {'avg_token_loss': avg_loss, 'ppl': ppl, 'bleu': bleu_score, 'total_eval_examples': len(hyps)}
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['lm','seq2seq'], required=True)
    parser.add_argument('--dataset', type=str, default='tiny_shakespeare')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--local_data_dir', type=str, default=None)
    parser.add_argument('--dataset_config', type=str, default=None)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learned_pos', action='store_true')
    parser.add_argument('--attention_type', type=str, default='full')
    parser.add_argument('--relative_pos', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=16)
    parser.add_argument('--local_window', type=int, default=8)
    parser.add_argument('--tokenizer_name', type=str, default='t5-small')
    parser.add_argument('--src_lang', type=str, default='en')
    parser.add_argument('--tgt_lang', type=str, default='de')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--output_file', type=str, default=None)

    args = parser.parse_args()
    # choose device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    # dispatch
    if args.task == 'lm':
        res = evaluate_lm(args, device)
    else:
        res = evaluate_seq2seq(args, device)
    print("Result:", res)

if __name__ == '__main__':
    main()
