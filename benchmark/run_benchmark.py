import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path

# Add the project root directory to Python's path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _merge_args(base: dict, override: dict) -> dict:
    out = dict(base or {})
    out.update(override or {})
    return out


def run_experiment(cfg: dict, exp: dict):
    output_root = Path(cfg.get("output_root", "runs"))
    exp_name = exp["name"]
    datasets = exp["datasets"]
    # resolve dataset paths
    ds_map = {d["name"]: d["path"] for d in cfg["datasets"]}
    data_dirs = [ds_map[name] for name in datasets]
    out_dir = output_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Algorithm selection (default: codet5)
    algorithm = exp.get("algorithm", cfg.get("algorithm", "codet5")).lower()
    base_ta = cfg.get("train_args", {})
    ta = _merge_args(base_ta, exp.get("train_args", {}))
    
    # Set environment variable for Python path to include the project root
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent) + ":" + env.get("PYTHONPATH", "")

    if algorithm == "codet5":
        model_name = exp.get("model") or cfg.get("model", {}).get("name", "Salesforce/codet5-small")
        cmd = [
            sys.executable, "training/train_codet5.py",
            "--data_dirs", *data_dirs,
            "--output_dir", str(out_dir),
            "--model_name", model_name,
            "--num_train_epochs", str(ta.get("num_train_epochs", 3)),
            "--per_device_train_batch_size", str(ta.get("per_device_train_batch_size", 8)),
            "--per_device_eval_batch_size", str(ta.get("per_device_eval_batch_size", 8)),
            "--gradient_accumulation_steps", str(ta.get("gradient_accumulation_steps", 1)),
            "--learning_rate", str(ta.get("learning_rate", 3e-4)),
            "--weight_decay", str(ta.get("weight_decay", 0.0)),
            "--warmup_steps", str(ta.get("warmup_steps", 1000)),
            "--max_source_length", str(ta.get("max_source_length", 1024)),
            "--max_target_length", str(ta.get("max_target_length", 8)),
            "--eval_top_k", str(ta.get("eval_top_k", 3)),
            "--seed", str(ta.get("seed", 42)),
        ]
    elif algorithm == "code2seq":
        cmd = [
            sys.executable, "training/train_code2seq.py",
            "--data_dirs", *data_dirs,
            "--output_dir", str(out_dir),
            "--num_train_epochs", str(ta.get("num_train_epochs", 3)),
            "--per_device_train_batch_size", str(ta.get("per_device_train_batch_size", 16)),
            "--per_device_eval_batch_size", str(ta.get("per_device_eval_batch_size", 16)),
            "--learning_rate", str(ta.get("learning_rate", 3e-4)),
            "--max_source_length", str(ta.get("max_source_length", 1024)),
            "--max_target_length", str(ta.get("max_target_length", 8)),
            "--eval_top_k", str(ta.get("eval_top_k", 3)),
            "--seed", str(ta.get("seed", 42)),
            "--d_model", str(ta.get("d_model", 512)),
            "--nhead", str(ta.get("nhead", 8)),
            "--num_layers", str(ta.get("num_layers", 3)),
            "--src_vocab_size", str(ta.get("src_vocab_size", 50000)),
            "--tgt_vocab_size", str(ta.get("tgt_vocab_size", 5000)),
        ]
    elif algorithm == "code2vec":
        cmd = [
            sys.executable, "training/train_code2vec.py",
            "--data_dirs", *data_dirs,
            "--output_dir", str(out_dir),
            "--num_train_epochs", str(ta.get("num_train_epochs", 3)),
            "--per_device_train_batch_size", str(ta.get("per_device_train_batch_size", 64)),
            "--per_device_eval_batch_size", str(ta.get("per_device_eval_batch_size", 64)),
            "--learning_rate", str(ta.get("learning_rate", 3e-4)),
            "--max_source_length", str(ta.get("max_source_length", 1024)),
            "--max_target_length", str(ta.get("max_target_length", 8)),
            "--eval_top_k", str(ta.get("eval_top_k", 3)),
            "--seed", str(ta.get("seed", 42)),
            "--src_vocab_size", str(ta.get("src_vocab_size", 50000)),
            "--tgt_vocab_size", str(ta.get("tgt_vocab_size", 5000)),
            "--hidden_dim", str(ta.get("hidden_dim", 512)),
        ]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"\nRunning experiment: {exp_name} ({algorithm})")
    print(" ".join(str(x) for x in cmd))
    
    subprocess.run(cmd, check=True, env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to benchmark/config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_root = Path(cfg.get("output_root", "runs"))
    output_root.mkdir(parents=True, exist_ok=True)

    for exp in cfg["experiments"]:
        run_experiment(cfg, exp)

    # Combine summary
    summary_lines = []
    summary_lines.append("# Benchmark Summary")
    summary_lines.append("")
    summary_lines.append("| Experiment | EM@1 | Top-k EM | P | R | F1 |")
    summary_lines.append("|---|---:|---:|---:|---:|---:|")
    for exp in cfg["experiments"]:
        exp_dir = output_root / exp["name"]
        res_file = exp_dir / "results.json"
        if not res_file.exists():
            continue
        import json
        with open(res_file, "r", encoding="utf-8") as f:
            res = json.load(f)
        o = res["overall"]
        summary_lines.append(f"| {exp['name']} | {o['exact_match@1']:.4f} | {o['topk_exact_match']:.4f} | {o['subtoken_precision']:.4f} | {o['subtoken_recall']:.4f} | {o['subtoken_f1']:.4f} |")

    with open(output_root / "benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print("Wrote", output_root / "benchmark_summary.md")

if __name__ == "__main__":
    main()