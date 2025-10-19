#!/usr/bin/env python3
# tools/extract_iwslt_test.py
"""
从本地 IWSLT 文件夹（例如 src/data/iwslt2017/en-de）抽取 test / dev 的 source 和 reference 文本
输出两个文件： <out_prefix>.src.txt 和 <out_prefix>.ref.txt
用法：
python tools/extract_iwslt_test.py --iwslt_dir src/data/iwslt2017/en-de --split test --src en --tgt de --out_prefix data/iwslt_test
"""
import os
import argparse
import re
import xml.etree.ElementTree as ET

def strip_tags(s):
    return re.sub(r"<[^>]+>", "", s).strip()

def read_seg(path):
    segs = []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        for seg in root.findall(".//seg"):
            txt = seg.text or ""
            if txt:
                segs.append(txt.strip())
    except Exception:
        # fallback
        txt = open(path, encoding="utf-8", errors="ignore").read()
        parts = re.findall(r"<seg[^>]*>(.*?)</seg>", txt, flags=re.DOTALL)
        for p in parts:
            t = strip_tags(p)
            if t:
                segs.append(t)
    return segs

def extract(iwslt_dir, split="test", src="en", tgt="de", out_prefix="data/iwslt_test"):
    assert os.path.isdir(iwslt_dir), f"{iwslt_dir} not exist"
    src_texts = []
    tgt_texts = []
    if split == "dev":
        en = os.path.join(iwslt_dir, f"IWSLT17.TED.dev2010.en-de.{src}.xml")
        de = os.path.join(iwslt_dir, f"IWSLT17.TED.dev2010.en-de.{tgt}.xml")
        src_texts = read_seg(en)
        tgt_texts = read_seg(de)
    else:
        # test: concat all tst*.en-de.*.xml pairs
        files = sorted(os.listdir(iwslt_dir))
        en_files = [f for f in files if f.startswith("IWSLT17.TED.tst") and f.endswith(f".{src}.xml")]
        de_files = [f for f in files if f.startswith("IWSLT17.TED.tst") and f.endswith(f".{tgt}.xml")]
        # pair by basename prefix
        de_map = {os.path.basename(p).rsplit(f".{tgt}.xml",1)[0]: p for p in de_files}
        for enf in sorted(en_files):
            key = enf.rsplit(f".{src}.xml",1)[0]
            if key in de_map:
                src_list = read_seg(os.path.join(iwslt_dir, enf))
                tgt_list = read_seg(os.path.join(iwslt_dir, de_map[key]))
                # best-effort pair
                L = min(len(src_list), len(tgt_list))
                src_texts.extend(src_list[:L])
                tgt_texts.extend(tgt_list[:L])
    # write out
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    src_out = out_prefix + ".src.txt"
    ref_out = out_prefix + ".ref.txt"
    with open(src_out, "w", encoding="utf-8") as f:
        for s in src_texts:
            f.write(s.replace("\n", " ") + "\n")
    with open(ref_out, "w", encoding="utf-8") as f:
        for r in tgt_texts:
            f.write(r.replace("\n", " ") + "\n")
    print(f"Wrote {len(src_texts)} examples to {src_out} and {ref_out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iwslt_dir", required=True)
    p.add_argument("--split", choices=["dev","test"], default="test")
    p.add_argument("--src", default="en")
    p.add_argument("--tgt", default="de")
    p.add_argument("--out_prefix", default="data/iwslt_test")
    args = p.parse_args()
    extract(args.iwslt_dir, args.split, args.src, args.tgt, args.out_prefix)
