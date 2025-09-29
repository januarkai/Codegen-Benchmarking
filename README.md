# Class Name Prediction Benchmark (Single-GPU)

This repo contains:
- A minimal miner to build a unified JSONL dataset:
  - Each row: {"lang","dataset","id","source","target"}
  - `source` is the class body text (with class name masked).
  - `target` is the class name as space-separated subtokens.
- Training/evaluation scripts for three model families:
  - CodeT5 (Seq2Seq transformer via Hugging Face)
  - Code2Seq (lightweight Transformer seq2seq implemented here)
  - Code2Vec-like (bag/sequence encoder with multi-label objective)
- A benchmark runner to compare multiple datasets and output a results table.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

GPU: Works on a single 8–12 GB GPU (adjust batch size/grad accumulation if needed).

## 2) Mine datasets

You can mine Java and/or Python codebases into a single JSONL file per dataset.

Examples:

```bash
# Java mining (recursively scans .java files)
python scripts/dataset_miner.py \
  --lang java \
  --input /path/to/java/repos_or_src \
  --dataset java_ds \
  --output data/java_ds/raw.jsonl

# Python mining (recursively scans .py files)
python scripts/dataset_miner.py \
  --lang python \
  --input /path/to/python/repos_or_src \
  --dataset py_ds \
  --output data/py_ds/raw.jsonl
```

Notes:
- The miner extracts each class, masks the true class name in the body to avoid leakage, and emits one JSONL record per class.
- Targets are split into subtokens (CamelCase and snake_case supported).

## 3) Split into train/val/test

```bash
# Java dataset split
python scripts/split_dataset.py \
  --input data/java_ds/raw.jsonl \
  --out_dir data/java_ds \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
  --seed 42

# Python dataset split
python scripts/split_dataset.py \
  --input data/py_ds/raw.jsonl \
  --out_dir data/py_ds \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
  --seed 42
```

This creates:
- data/<dataset>/{train.jsonl, valid.jsonl, test.jsonl}

## 4) Quick single runs

Train on a dataset and evaluate per-dataset (each accepts multiple `--data_dirs`):

```bash
# CodeT5-small
python training/train_codet5.py \
  --data_dirs data/java_ds \
  --output_dir runs/java_only \
  --model_name Salesforce/codet5-small \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-4 \
  --max_source_length 1024 \
  --max_target_length 8 \
  --eval_top_k 3

# Code2Seq
python training/train_code2seq.py \
  --data_dirs data/java_ds \
  --output_dir runs/java_only_code2seq \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-4 \
  --max_source_length 1024 \
  --max_target_length 8 \
  --eval_top_k 3 \
  --d_model 512 --nhead 8 --num_layers 3

# Code2Vec-like
python training/train_code2vec.py \
  --data_dirs data/java_ds \
  --output_dir runs/java_only_code2vec \
  --num_train_epochs 3 \
  --per_device_train_batch_size 64 \
  --learning_rate 3e-4 \
  --max_source_length 1024 \
  --max_target_length 8 \
  --eval_top_k 3 \
  --hidden_dim 512
```

Outputs:
- Model checkpoints and tokenizer.
- results.json and results.md with overall and per-dataset metrics:
  - Exact Match (top-1)
  - Subtoken Precision/Recall/F1 (top-1)
  - Top-k Exact Match (k as configured)

## 5) Run the benchmark suite (multi-dataset / multi-algorithm)

Edit `benchmark/config.yaml` to point to your datasets and desired experiments. Each experiment can set `algorithm: {codet5|code2seq|code2vec}` and optional `train_args` overrides. Then:

```bash
python benchmark/run_benchmark.py --config benchmark/config.yaml
```

Results per experiment appear under `runs/<experiment_name>/results.md` and a combined report: `runs/benchmark_summary.md`.

## Tips

- If you get OOM:
  - Lower `--per_device_train_batch_size`
  - Increase `--gradient_accumulation_steps`
  - Reduce `--max_source_length`
- Ensure class names are masked in sources (the miner does this for you).
- Add more datasets by repeating the mining and splitting steps, then list them in the benchmark config.
 - For `code2seq`/`code2vec`, vocabularies are built on the train split; adjust `src_vocab_size`/`tgt_vocab_size` in `train_args` if needed.
