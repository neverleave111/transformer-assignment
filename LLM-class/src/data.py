# src/data.py
import os
import re
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

# Check for HF datasets optional usage
try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

DEFAULT_TOKENIZER = "t5-small"

def _strip_xml_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()

def _read_train_tags(dirpath: str, src: str = "en", tgt: str = "de") -> Tuple[List[str], List[str]]:
    en_file = os.path.join(dirpath, f"train.tags.{src}-{tgt}.{src}")
    de_file = os.path.join(dirpath, f"train.tags.{src}-{tgt}.{tgt}")
    if not (os.path.exists(en_file) and os.path.exists(de_file)):
        raise FileNotFoundError(f"Expected train tags files not found in {dirpath}: {en_file}, {de_file}")
    src_lines, tgt_lines = [], []
    with open(en_file, "r", encoding="utf-8", errors="ignore") as f_en, \
         open(de_file, "r", encoding="utf-8", errors="ignore") as f_de:
        for a, b in zip(f_en, f_de):
            a = _strip_xml_tags(a)
            b = _strip_xml_tags(b)
            if a == "" or b == "":
                continue
            src_lines.append(a)
            tgt_lines.append(b)
    return src_lines, tgt_lines

def _read_xml_seg_file(path: str) -> List[str]:
    texts = []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        for seg in root.findall(".//seg"):
            if seg is None:
                continue
            txt = seg.text or ""
            txt = txt.strip()
            if txt:
                texts.append(txt)
    except ET.ParseError:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            parts = re.findall(r"<seg[^>]*>(.*?)</seg>", content, flags=re.DOTALL)
            for p in parts:
                t = _strip_xml_tags(p)
                if t:
                    texts.append(t)
    return texts

def _read_dev_or_test(dirpath: str, src: str="en", tgt:str="de", which:str="dev") -> Tuple[List[str], List[str]]:
    src_texts, tgt_texts = [], []
    if which == "dev":
        en_path = os.path.join(dirpath, "IWSLT17.TED.dev2010.en-de." + src + ".xml")
        de_path = os.path.join(dirpath, "IWSLT17.TED.dev2010.en-de." + tgt + ".xml")
        if not (os.path.exists(en_path) and os.path.exists(de_path)):
            raise FileNotFoundError(f"Dev files not found: {en_path} or {de_path}")
        src_texts = _read_xml_seg_file(en_path)
        tgt_texts = _read_xml_seg_file(de_path)
    elif which == "test":
        en_files = sorted([os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.startswith("IWSLT17.TED.tst") and f.endswith(f".{src}.xml")])
        de_files = sorted([os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.startswith("IWSLT17.TED.tst") and f.endswith(f".{tgt}.xml")])
        if len(en_files) == 0 or len(de_files) == 0:
            raise FileNotFoundError("No test xml files found for tst* in local dir.")
        de_map = {os.path.basename(p).rsplit(f".{tgt}.xml",1)[0]: p for p in de_files}
        for enf in en_files:
            key = os.path.basename(enf).rsplit(f".{src}.xml",1)[0]
            if key in de_map:
                src_list = _read_xml_seg_file(enf)
                tgt_list = _read_xml_seg_file(de_map[key])
                src_texts.extend(src_list)
                tgt_texts.extend(tgt_list)
    else:
        raise ValueError("which must be 'dev' or 'test'")
    L = min(len(src_texts), len(tgt_texts))
    return src_texts[:L], tgt_texts[:L]

def load_iwslt_local(dirpath: str, src: str="en", tgt: str="de", split: str="train") -> Tuple[List[str], List[str]]:
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"Local IWSLT dir not found: {dirpath}")
    split = split.lower()
    if split == "train":
        return _read_train_tags(dirpath, src=src, tgt=tgt)
    elif split == "dev":
        return _read_dev_or_test(dirpath, src=src, tgt=tgt, which="dev")
    elif split == "test":
        return _read_dev_or_test(dirpath, src=src, tgt=tgt, which="test")
    else:
        raise ValueError("split must be one of train/dev/test")

# ----------------------------
# build_seq2seq_dataloader
# ----------------------------
def build_seq2seq_dataloader(dataset_name: str,
                             tokenizer_name: str = DEFAULT_TOKENIZER,
                             seq_len: int = 128,
                             batch_size: int = 16,
                             split: str = "train",
                             src_lang: Optional[str] = None,
                             tgt_lang: Optional[str] = None,
                             dataset_config: Optional[str] = None,
                             max_samples: Optional[int] = None,
                             shuffle: bool = True,
                             dataset_local_dir: Optional[str] = None,
                             use_fast_tokenizer: bool = True):
    name = dataset_name.lower()
    if name == "iwslt2017" and dataset_local_dir is not None and os.path.isdir(dataset_local_dir):
        print(f"Loading local IWSLT2017 from {dataset_local_dir} split={split}")
        src_texts, tgt_texts = load_iwslt_local(dataset_local_dir, src=(src_lang or "en"), tgt=(tgt_lang or "de"), split=split)
    else:
        if not _HAS_DATASETS:
            raise RuntimeError("datasets library not available and local data not provided.")
        if name == "iwslt2017":
            ds_name = "iwslt2017"
            config = dataset_config or (f"{src_lang}-{tgt_lang}" if src_lang and tgt_lang else None)
            raw = load_dataset(ds_name, config, split=split) if config else load_dataset(ds_name, split=split)
        elif name in ("ted", "ted_talks", "ted_talks_iwslt"):
            ds_name = "ted_talks_iwslt"
            config = dataset_config or (f"{src_lang}-{tgt_lang}" if src_lang and tgt_lang else None)
            raw = load_dataset(ds_name, config, split=split) if config else load_dataset(ds_name, split=split)
        elif name == "gigaword":
            raw = load_dataset("gigaword", split=split)
        elif name in ("cnn","cnn_dailymail","cnn-dailymail"):
            cfg = dataset_config or "3.0.0"
            raw = load_dataset("cnn_dailymail", cfg, split=split)
        else:
            raise ValueError(f"Unsupported dataset_name {dataset_name}")
        src_texts, tgt_texts = [], []
        for ex in raw:
            if "translation" in ex and isinstance(ex["translation"], dict):
                tr = ex["translation"]
                if src_lang and tgt_lang and src_lang in tr and tgt_lang in tr:
                    s = tr[src_lang]; t = tr[tgt_lang]
                else:
                    keys = list(tr.keys())
                    s = tr[keys[0]]; t = tr[keys[1]]
            elif "article" in ex and "highlights" in ex:
                s = ex["article"]; t = ex["highlights"]
            elif "document" in ex and "summary" in ex:
                s = ex["document"]; t = ex["summary"]
            elif "src" in ex and "tgt" in ex:
                s = ex["src"]; t = ex["tgt"]
            else:
                strs = [v for v in ex.values() if isinstance(v, str)]
                if len(strs) >= 2:
                    s,t = strs[0], strs[1]
                else:
                    continue
            src_texts.append(s)
            tgt_texts.append(t)

    if max_samples is not None:
        N = min(len(src_texts), max_samples)
        src_texts = src_texts[:N]
        tgt_texts = tgt_texts[:N]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    enc = tokenizer(src_texts, truncation=True, padding="max_length", max_length=seq_len, return_tensors="pt")
    # use text_target if tokenizer supports (avoid deprecation)
    try:
        lab = tokenizer(text_target=tgt_texts, truncation=True, padding="max_length", max_length=seq_len, return_tensors="pt")
    except TypeError:
        # fallback for older transformers
        with tokenizer.as_target_tokenizer():
            lab = tokenizer(tgt_texts, truncation=True, padding="max_length", max_length=seq_len, return_tensors="pt")

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    label_ids = lab["input_ids"]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    label_ids[label_ids == pad_id] = -100

    dataset = TensorDataset(input_ids, attention_mask, label_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"Prepared dataloader: {len(dataset)} examples, batch_size={batch_size}, seq_len={seq_len}")
    return dataloader, tokenizer

# ----------------------------
# build_char_dataset_from_texts (fallback small dataset used in training script)
# ----------------------------
def build_char_dataset_from_texts(texts, seq_len=128, batch_size=64, stride=1, shuffle=True):
    raw = "\n".join(texts)
    chars = sorted(list(set(raw)))
    if len(chars) == 0:
        chars = [chr(i) for i in range(32,127)]
    stoi = {c:i for i,c in enumerate(chars)}
    itos = {i:c for i,c in enumerate(chars)}
    ids = [stoi.get(c,0) for c in raw]
    X, Y = [], []
    if len(ids) <= seq_len:
        # pad/repeat so we have at least one window
        ids = ids + ids[:(seq_len+1)]
    for i in range(0, max(1, len(ids)-seq_len), stride):
        X.append(ids[i:i+seq_len])
        Y.append(ids[i+1:i+seq_len+1])
    X = np.array(X, dtype=np.int64)
    Y = np.array(Y, dtype=np.int64)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # simple tokenizer-like object for compatibility
    class CharTok:
        def __init__(self, stoi, itos):
            self._stoi = stoi
            self._itos = itos
        def encode(self, s):
            return [self._stoi.get(c,0) for c in s]
        def decode(self, ids):
            return "".join(self._itos.get(i,'?') for i in ids)
        def vocab_size(self):
            return len(self._stoi)
    tokenizer = CharTok(stoi, itos)
    return loader, tokenizer
