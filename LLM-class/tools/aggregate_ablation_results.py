#!/usr/bin/env python3
# tools/aggregate_ablation_results.py
import os, re, csv, json
import matplotlib.pyplot as plt

ROOT = "results/ablation"
out_csv = "results/ablation_summary.csv"
rows = []
pattern = re.compile(r"Epoch\s+(\d+)\s+finished\s+—\s+epoch avg loss:\s*([0-9eE\.\-]+)")

for run in sorted(os.listdir(ROOT)):
    run_dir = os.path.join(ROOT, run)
    logf = os.path.join(run_dir, "log.txt")
    if not os.path.exists(logf):
        continue
    with open(logf, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    matches = pattern.findall(txt)
    if matches:
        last_epoch, last_loss = matches[-1]
        rows.append((run, int(last_epoch), float(last_loss)))
    else:
        # try to find "Training finished" and maybe a printed avg
        m = re.search(r"Training finished.*?Saved curve to.*", txt, re.S)
        rows.append((run, None, None))

# write csv
os.makedirs("results", exist_ok=True)
with open(out_csv, "w", newline="", encoding="utf-8") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["run", "last_epoch", "last_epoch_avg_loss"])
    for r in rows:
        writer.writerow(r)

# plot bar chart excluding None
runs = [r for r,e,l in rows if l is not None]
losses = [l for r,e,l in rows if l is not None]
plt.figure(figsize=(max(6, len(runs)*0.6),4))
plt.bar(range(len(runs)), losses)
plt.xticks(range(len(runs)), runs, rotation=45, ha='right')
plt.ylabel("Final epoch average loss")
plt.tight_layout()
plt.savefig("results/ablation_summary_loss.png")
print("Wrote", out_csv, "and results/ablation_summary_loss.png")
