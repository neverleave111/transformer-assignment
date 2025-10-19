# src/train.py
import os
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# try both import styles so script works whether executed as `python src/train.py`
# (script dir is src) or as module (`python -m src.train`)
try:
    from data import build_seq2seq_dataloader, build_char_dataset_from_texts
except Exception:
    from src.data import build_seq2seq_dataloader, build_char_dataset_from_texts

# model imports (Encoder/Decoder implemented in src/model.py)
try:
    from model import Encoder, Decoder
except Exception:
    from src.model import Encoder, Decoder

# utils
try:
    from utils import set_seed, count_parameters
except Exception:
    from src.utils import set_seed, count_parameters


def save_checkpoint(path, ckpt):
    torch.save(ckpt, path)


def train_tiny_shakespeare(args, device):
    """
    Train a character-level language model on tiny_shakespeare.
    Uses Encoder (embedding+pos+stack) + a linear lm_head mapping embed_dim -> vocab_size.
    DataLoader yields (X, Y) where X,Y are LongTensors (B, T).
    """
    # try to load HF dataset if possible, else fallback to builtin
    texts = None
    if args.use_hf_datasets:
        try:
            from datasets import load_dataset
            ds = load_dataset("tiny_shakespeare")["train"]
            texts = ds["text"]
            print("Loaded tiny_shakespeare from HF datasets")
        except Exception as e:
            print("Could not load tiny_shakespeare from HF hub (falling back to builtin). Error:", e)

    if texts is None:
        # builtin tiny fallback (enough to run small experiments)
        print("Using builtin tiny_shakespeare fallback (offline).")
        texts = [
            "First Citizen: Before we proceed any further, hear me speak.",
            "All: Speak, speak.",
            "First Citizen: You are all resolved rather to die than to famish?",
            "All: Resolved. resolved.",
            "KING: You shall do well."
        ] * 200  # replicate to get some data

    loader, tokenizer = build_char_dataset_from_texts(texts, seq_len=args.seq_len, batch_size=args.batch_size, stride=1, shuffle=True)
    vocab_size = tokenizer.vocab_size()

    # build model: encoder + lm head
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
    lm_head = nn.Linear(args.embed_dim, vocab_size)
    if args.no_pos:
    # 用 Identity 替换位置编码
    # 有些实现里可能把 pos 字段命名为 'pos'（此处与当前 model.py 保持一致）
        encoder.pos = nn.Identity()
        print("Warning: positional encodings removed for encoder (encoder.pos replaced with Identity).")
    encoder = encoder.to(device)
    lm_head = lm_head.to(device)

    params = list(encoder.parameters()) + list(lm_head.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * (len(loader) if len(loader) > 0 else 1)
    def lr_lambda(step):
        warmup = max(1, int(0.1 * max(1, total_steps)))
        if step < warmup:
            return float(step) / float(max(1, warmup))
        return max(0.0, float(max(1, total_steps) - step) / float(max(1, max(1, total_steps) - warmup)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    print("Parameter count:", count_parameters(encoder) + sum(p.numel() for p in lm_head.parameters() if p.requires_grad))

    train_losses = []
    epoch_avgs = []
    global_step = 0
    encoder.train(); lm_head.train()

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        running_loss = 0.0; running_count = 0; epoch_total = 0.0; epoch_steps = 0

        for xb, yb in pbar:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            enc_out, _ = encoder(xb)   # (B, T, embed_dim)
            logits = lm_head(enc_out)  # (B, T, V)
            B, T, V = logits.size()
            loss = criterion(logits.view(B*T, V), yb.view(B*T))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            step_loss = loss.item()
            running_loss += step_loss; running_count += 1
            epoch_total += step_loss; epoch_steps += 1

            if global_step % args.log_every == 0:
                avg = running_loss / running_count if running_count > 0 else 0.0
                train_losses.append(avg)
                pbar.set_postfix({'loss': f'{avg:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
                running_loss = 0.0; running_count = 0

        if running_count > 0:
            avg_left = running_loss / running_count
            train_losses.append(avg_left)
            pbar.set_postfix({'loss': f'{avg_left:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        epoch_avg = epoch_total / epoch_steps if epoch_steps > 0 else 0.0
        epoch_avgs.append(epoch_avg)
        print(f"Epoch {epoch+1} finished — epoch avg loss: {epoch_avg:.4f}")

        # save checkpoint
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt = {
            'encoder_state_dict': encoder.state_dict(),
            'lm_head_state_dict': lm_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'tokenizer': getattr(tokenizer, "__dict__", None),
            'args': vars(args)
        }
        if epoch==49:
            save_checkpoint(os.path.join(args.save_dir, f'ckpt_epoch{epoch+1}.pth'), ckpt)

    if len(train_losses) == 0 and len(epoch_avgs) > 0:
        train_losses = epoch_avgs

    plt.figure()
    plt.plot(train_losses, marker='o')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'train_curve.png'))
    print('Training finished. Saved curve to', os.path.join(args.save_dir, 'train_curve.png'))


def train_seq2seq(args, device):
    """
    Train a simple encoder-decoder pipeline using build_seq2seq_dataloader
    Supports args.enc_only: if True, train encoder only by feeding target tokens into encoder
    (i.e., treat source==target for ablation).
    """
    # prepare dataloader & tokenizer
    loader, tokenizer = build_seq2seq_dataloader(args.dataset,
                                                tokenizer_name=args.tokenizer_name,
                                                seq_len=args.seq_len,
                                                batch_size=args.batch_size,
                                                split='train',
                                                src_lang=args.src_lang,
                                                tgt_lang=args.tgt_lang,
                                                dataset_config=args.dataset_config,
                                                max_samples=args.max_samples,
                                                shuffle=True,
                                                dataset_local_dir=args.local_data_dir)
    # obtain vocab size from tokenizer (handle different tokenizer APIs)
    try:
        vocab_size = tokenizer.vocab_size if isinstance(tokenizer.vocab_size, int) else tokenizer.vocab_size()
    except Exception:
        vocab_size = getattr(tokenizer, "vocab_size", None) or getattr(tokenizer, "model_max_length", 32000)
    # build encoder (and decoder unless enc_only)
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

    # if we want encoder-only ablation, create a lm_head to map encoder outputs to vocab
    if args.enc_only:
        lm_head = torch.nn.Linear(args.embed_dim, vocab_size)
        lm_head = lm_head.to(device)
    else:
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
        decoder = decoder.to(device)

    encoder = encoder.to(device)

    # replace pos enc if requested
    if args.no_pos:
        import torch.nn as nn
        encoder.pos = nn.Identity()
        if not args.enc_only:
            decoder.pos = nn.Identity()

    # collect params
    if args.enc_only:
        params = list(encoder.parameters()) + list(lm_head.parameters())
    else:
        params = list(encoder.parameters()) + list(decoder.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * (len(loader) if len(loader) > 0 else 1)
    def lr_lambda(step):
        warmup = max(1, int(0.1 * max(1, total_steps)))
        if step < warmup:
            return float(step) / float(max(1, warmup))
        return max(0.0, float(max(1, total_steps) - step) / float(max(1, max(1, total_steps) - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # loss: use ignore_index=-100 since dataloader uses -100 for label padding
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    print("Parameter count (encoder{}):".format(" + lm_head" if args.enc_only else " + decoder"), count_parameters(encoder) + (sum(p.numel() for p in (lm_head.parameters() if args.enc_only else decoder.parameters()) if p.requires_grad)))

    train_losses = []
    epoch_avgs = []
    global_step = 0
    encoder.train()
    if not args.enc_only:
        decoder.train()
    else:
        lm_head.train()

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        running_loss = 0.0; running_count = 0; epoch_total = 0.0; epoch_steps = 0

        for batch in pbar:
            optimizer.zero_grad()
            # batch is (input_ids, attention_mask, label_ids)
            input_ids, attention_mask, label_ids = [t.to(device) for t in batch]

            if args.enc_only:
                # For ablation: treat target as encoder input (source==target).
                # label_ids has -100 for pad; create encoder_input by replacing -100 with pad id (0)
                enc_input = label_ids.clone()
                enc_input[enc_input == -100] = 0  # assume pad idx 0 is safe; tokenizer.pad_token_id used elsewhere
                enc_out, _ = encoder(enc_input)  # (B, T, embed_dim)
                logits = lm_head(enc_out)  # (B, T, V)
                B, T, V = logits.size()
                loss = criterion(logits.view(B*T, V), label_ids.view(B*T))
            else:
                # standard encoder-decoder training using teacher forcing with label_ids as decoder input
                enc_out, _ = encoder(input_ids)
                dec_input = label_ids.clone()
                dec_input[dec_input == -100] = 0
                logits, _, _ = decoder(dec_input, enc_out)
                B, T, V = logits.size()
                loss = criterion(logits.view(B*T, V), label_ids.view(B*T))

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            step_loss = loss.item()
            running_loss += step_loss; running_count += 1
            epoch_total += step_loss; epoch_steps += 1

            if global_step % args.log_every == 0:
                avg = running_loss / running_count if running_count>0 else 0.0
                train_losses.append(avg)
                pbar.set_postfix({'loss': f'{avg:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
                running_loss = 0.0; running_count = 0

        if running_count > 0:
            avg_left = running_loss / running_count
            train_losses.append(avg_left)
            pbar.set_postfix({'loss': f'{avg_left:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        epoch_avg = epoch_total / epoch_steps if epoch_steps>0 else 0.0
        epoch_avgs.append(epoch_avg)
        print(f"Epoch {epoch+1} finished — epoch avg loss: {epoch_avg:.4f}")

        # save checkpoint
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt = {
            "encoder_state": encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args)
        }
        if args.enc_only:
            ckpt["lm_head_state"] = lm_head.state_dict()
        else:
            ckpt["decoder_state"] = decoder.state_dict()
        if (epoch+1)%10==0:
            torch.save(ckpt, os.path.join(args.save_dir, f"ckpt_epoch{epoch+1}.pth"))

    if len(train_losses) == 0 and len(epoch_avgs) > 0:
        train_losses = epoch_avgs

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_losses, marker='o')
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, "train_curve.png"))
    print("Saved curve to", os.path.join(args.save_dir, "train_curve.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tiny_shakespeare',
                        help='tiny_shakespeare | iwslt2017 | cnn_dailymail | gigaword | ted')
    parser.add_argument('--dataset_config', type=str, default=None)
    parser.add_argument('--local_data_dir', type=str, default=None, help='local dataset dir (e.g. iwslt local dir)')
    parser.add_argument('--src_lang', type=str, default='en')
    parser.add_argument('--tgt_lang', type=str, default='de')
    parser.add_argument('--tokenizer_name', type=str, default='t5-small')
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learned_pos', action='store_true')
    parser.add_argument('--attention_type', type=str, default='full', choices=['full','linear','local'])
    parser.add_argument('--relative_pos', action='store_true')
    parser.add_argument('--enc_only', action='store_true', help='train encoder only (treat target as encoder input; used for ablation)')
    parser.add_argument('--max_relative_position', type=int, default=16)
    parser.add_argument('--local_window', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--use_hf_datasets', action='store_true', help='allow loading tiny_shakespeare from HF datasets if available')
    parser.add_argument('--no_pos', action='store_true', help='remove positional encodings')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    if args.dataset == 'tiny_shakespeare':
        train_tiny_shakespeare(args, device)
    else:
        train_seq2seq(args, device)


if __name__ == '__main__':
    main()
