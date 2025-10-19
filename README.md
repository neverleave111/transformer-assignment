## 概要

本仓库为手工实现的 Transformer（含多种注意力变体、位置编码选项、encoder/decoder、训练稳定性技巧等），用于在小规模数据集上做消融实验与模型评估。主要任务与脚本见 `scripts/` 与 `scripts/ablation/`。

---

## 环境与依赖（推荐）

下面给出**优先推荐（conda）**的安装步骤，后面还提供 **pip 备用方案** 与常见错误的排查方法。

### 1) 创建 conda 环境（推荐）

```bash
# 创建并激活环境
conda create -n llm_class python=3.9 -y
conda activate llm_class
```

### 2) 安装 PyTorch（按你的 CUDA 版本选择）

* 如果你 **没有 GPU（CPU only）**：

```bash
conda install -c pytorch pytorch cpuonly -y
```

* 如果你 **有 GPU**（示例：CUDA 11.8，请根据实际 CUDA 版本替换）：

```bash
conda install -c pytorch pytorch pytorch-cuda=11.8 -c nvidia -y
```

> 如需其他 CUDA 版本，请参照 PyTorch 官方安装说明选择合适 `pytorch-cuda` 包。

### 3) 安装 transformers / datasets / sentencepiece / tokenizers / safetensors（首选 conda-forge）

这些包在 conda-forge 上通常有预编译二进制，能避免编译失败：

```bash
conda install -c conda-forge transformers datasets sentencepiece tokenizers safetensors regex -y
```

### 4) 其它常用 Python 包

```bash
conda install -c conda-forge matplotlib tqdm pandas sacrebleu rouge_score -y
```

### 5) 如果必须使用 pip（备用）

先升级 pip / setuptools / wheel，然后安装：

```bash
pip install --upgrade pip setuptools wheel
pip install transformers datasets sentencepiece tokenizers safetensors regex
pip install matplotlib tqdm pandas sacrebleu rouge_score
```

> **注意**：在 Windows + pip 的组合下，`tokenizers` / `safetensors` 有时会尝试从源码编译，可能需要 Rust / C 编译工具链。若出现编译错误，建议切换到 conda-forge 或使用 WSL / Linux。

---

## 常见安装问题与解决方法

* **pip 报错 `subprocess-exited-with-error`（safetensors / tokenizers）**
  解决：优先使用 `conda install -c conda-forge tokenizers safetensors`。若必须用 pip，请确保系统已安装 Rust toolchain（`rustup`）、并升级 `pip`/`setuptools`/`wheel`。

* **警告 `Ignoring invalid distribution -ip` / `-orch`**
  这是 pip/conda 安装历史遗留警告，通常不影响运行，但可尝试清理环境或重建虚拟环境。

* **datasets 无法从 Hub 下载（ConnectionError）**
  在无法联网环境下：手动下载数据并用 `--local_data_dir` 指定本地路径；或用脚本 `tools/extract_iwslt_test.py` 处理本地 XML。

---

## 数据准备（数据仓库中已经全部包含）

### tiny_shakespeare

* 无需额外下载（脚本/代码中通常能自动构造或内置）。

### IWSLT2017 (en-de)

* 如果你的环境不能自动下载，请把 IWSLT 的 XML 文件手动放到：

```
src/data/iwslt2017/en-de/
```

目录下应包含诸如 `IWSLT17.TED.tst2010.en-de.en.xml`、`train.tags.en-de.en` 等文件。训练脚本支持 `--local_data_dir` 指定路径。

### CNN/DailyMail

* 如果使用该数据集且无法联网，请提前准备好相应的本地数据，并在 `src/data.py` 中适配读取逻辑或在调用时指定本地路径。

---

## 复现实验：关键命令（示例）

下面是针对常用场景的**完整命令范例**，请在终端（并已激活 `llm_class` 环境）中运行。

> 注意：命令示例中的 `--save_dir` / `--local_data_dir` / `--dataset` 等参数应与 `src/train.py` 的 argparse 定义一致。若你修改了 `train.py` 参数名，请同步修改下面命令。

### A. 训练 tiny_shakespeare（baseline）

```bash
python src/train.py \
  --dataset tiny_shakespeare \
  --epochs 10 \
  --batch_size 64 \
  --seq_len 128 \
  --embed_dim 128 \
  --num_layers 2 \
  --num_heads 4 \
  --ff_dim 512 \
  --lr 3e-4 \
  --grad_clip 1.0 \
  --save_dir results/tiny_baseline \
  --log_every 10
```

### B. 运行局部注意力（local attention，window=8）

```bash
python src/train.py \
  --dataset tiny_shakespeare \
  --attention_type local \
  --local_window 8 \
  --epochs 10 \
  --batch_size 64 \
  --save_dir results/attn_local_w8
```

### C. 训练 IWSLT (encoder-decoder)，使用本地 IWSLT 文件

```bash
python src/train.py \
  --dataset iwslt2017 \
  --local_data_dir src/data/iwslt2017/en-de \
  --src_lang en \
  --tgt_lang de \
  --seq_len 128 \
  --batch_size 32 \
  --epochs 50 \
  --save_dir results/iwslt_encdec
```

### D. 同时启用相对位置偏置的 IWSLT 训练（示例）

```bash
python src/train.py \
  --dataset iwslt2017 \
  --local_data_dir src/data/iwslt2017/en-de \
  --src_lang en \
  --tgt_lang de \
  --relative_pos \
  --max_relative_position 32 \
  --seq_len 128 \
  --batch_size 32 \
  --epochs 50 \
  --save_dir results/iwslt_encdec_rel
```

### E. 运行脚本（若你更愿意用仓库内脚本）

仓库 `scripts/` 下已有调用示例，你可直接运行（先确保脚本有可执行权限）：

```bash
bash scripts/run_tiny.sh
bash scripts/run_local_attention.sh
bash scripts/run_iwslt_en_de.sh
```

### F. 批量消融（scripts/ablation）

运行预设消融集合（注意：会占用较多时间与资源）：

```bash
bash scripts/ablation/run_baselines.sh
```

---

## 生成预测与评估（示例）

训练完 checkpoint 后，使用工具脚本对 test 集生成预测（贪心/beam）并保存：

```bash
python tools/generate_and_eval.py \
  --ckpt results/iwslt_encdec/ckpt_epoch50.pth \
  --inputs data/iwslt_test.src.txt \
  --refs data/iwslt_test.ref.txt \
  --out results/iwslt_encdec/preds.greedy.txt \
  --tokenizer t5-small \
  --device cuda \
  --mode greedy \
  --max_len 80
```

生成后，你可以用 sacrebleu / rouge 等工具（或脚本内部已有评估）进行质量评估。

---

## 可视化训练曲线（流程说明）

如果训练脚本把每 epoch 的 loss 保存为可解析格式（如 `results/<run>/train_losses.npy` 或 `train_log.txt`），你可以用仓库内或自定义脚本绘制训练/验证曲线，并保存为 `results/<run>/train_curve.png`。这样 `report.tex` 中的占位图片就能被替换为真实曲线。

---

## 结果保存约定

建议每次 run 用独立的 `--save_dir results/<run_name>`，在该目录下保存：

* checkpoint 文件（例如 `ckpt_epoch50.pth`）
* 训练日志（epoch / step loss）
* 训练曲线图片（PNG）
* 解码预测（`preds.greedy.txt` / `preds.beam.txt`）
* 评估结果（JSON 或文本）

---

## 常见问题（运行时）与排查建议

* **训练时报错 `ModuleNotFoundError: No module named 'datasets'`**

  * 检查 `datasets` 是否已安装；推荐 `conda install -c conda-forge datasets`。

* **无法访问 Hugging Face Hub（下载数据失败）**

  * 在无法联网的环境中，手动准备数据并通过 `--local_data_dir` 指定路径。

* **local attention / mask shape mismatch（RuntimeError: size mismatch）**

  * 排查 q/k/v 的 reshape 与 mask 的维度，确认 multi-head reshape（batch, heads, seq_len, head_dim）一致；打印临时 shape 有助定位。

* **pip 安装 safetensors/tokenizers 失败**

  * 首选 `conda install -c conda-forge tokenizers safetensors`；如必须用 pip，请确保 Rust toolchain 已安装（仅在源码构建时需要）。

---

## 推荐硬件与时间估算（粗略）

* tiny_shakespeare（小模型）：单次 run（10 epochs）在中端 GPU（例如 RTX 2080/RTX 3060）可能在数分钟到数十分钟。
* IWSLT seq2seq（中等模型，50 epochs）：在单张 GPU 上可能需要数小时到十小时不等。
* 建议至少使用带 12GB 以上显存的 GPU（如 RTX 3080），若做大规模实验建议使用更高端卡或多卡训练。

---

## 目录结构（简述）

```
src/          # 模型、训练、数据处理
scripts/      # 运行与实验脚本
scripts/ablation/  # 消融脚本集合
tools/        # 生成与评估工具
results/      # 实验输出
report.tex    # LaTeX 实验报告
README.md
```


