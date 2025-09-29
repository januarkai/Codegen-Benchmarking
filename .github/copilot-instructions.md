# Copilot instructions for this repo

Purpose:## Tuning & troubleshooting
- OOM: lower `--per_device_train_batch_size`, raise `--gradient_accumulation_steps`, reduce `--max_source_length`.
- Java mining requires `javalang`; install errors -> skip Java or install it.
- Quick debug loop: run 1 epoch with small `max_source_length` (e.g., 256) and small batch sizes; verify `results.json` and `predictions.jsonl` exist under `runs/<exp>`.
- Import errors: If you get `ModuleNotFoundError: No module named 'scripts'`, you need to ensure the project root is in Python's path. Fix this by:
  - Always run commands from the project root directory
  - Or add `import sys; sys.path.insert(0, str(Path(__file__).parent.parent))` at the top of the main script
  - Or use `PYTHONPATH=. python benchmark/run_benchmark.py --config benchmark/config.yaml`gle-GPU benchmark for predicting class names from class bodies with multiple algorithms (CodeT5, Code2Seq, Code2Vec-like). Pipeline: mine -> split -> train/eval -> aggregate.

# Copilot instructions for this repo
## Big picture
- Data schema (JSONL): {"lang","dataset","id","source","target"}. See `scripts/utils/dataset_miner.py`.
- Target = class name subtokens (lowercase, space-separated) via `scripts/utils/subtokenize.py`.
- Leakage prevention: occurrences of the true class name in `source` are masked with `CLASSTOKEN` (MASK_TOKEN).

Results schema contract (all algorithms must emit the same shape)
	- `overall`: {`exact_match@1`, `topk_exact_match`, `topk`, `subtoken_precision`, `subtoken_recall`, `subtoken_f1`, `count`}
	- `per_dataset`: map of dataset -> same metrics + `count`
	- `config` (training args), `num_examples`, and `algorithm`

Key workflows (see README for commands)

Outputs & metrics
- Dataset dir layout is fixed: `train.jsonl`, `valid.jsonl`, `test.jsonl` (hard-coded in loader).
- JSONL must include exactly: `source` (masked body text) and `target` (lowercase subtokens). Extra fields are dropped during tokenization.

Training specifics to know
- Multiple datasets are concatenated via HF Datasets; per-dataset metrics use the preserved `dataset` field from raw rows.

Tuning & troubleshooting
- OOM: lower `--per_device_train_batch_size`, raise `--gradient_accumulation_steps`, reduce `--max_source_length`.
- Java mining requires `javalang`; install errors -> skip Java or install it.
 - Quick debug loop: run 1 epoch with small `max_source_length` (e.g., 256) and small batch sizes; verify `results.json` and `predictions.jsonl` exist under `runs/<exp>`.

Where to change things
- Datasets/experiments/knobs: edit `benchmark/config.yaml` (set `algorithm`, tweak `train_args`; forwarded to the chosen training script).
- Tokenization/masking rules: `scripts/utils/subtokenize.py` and `MASK_TOKEN` in `dataset_miner.py`.
- Add a language: extend `dataset_miner.py` with extractor; keep masking + subtokenization consistent.

Reference files
- Quickstart: `README.md`
- Mining/splitting/tokenization: `scripts/utils/{dataset_miner.py,split_dataset.py,subtokenize.py}`
- Training/eval: `training/train_codet5.py`
- Benchmark: `benchmark/run_benchmark.py`, `benchmark/config.yaml`
- Deps: `requirements.txt`
