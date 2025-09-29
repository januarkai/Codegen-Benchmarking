import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add the project root to Python's path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

from scripts.utils.subtokenize import subtokenize_identifier

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_multi_dataset_dirs(dirs: List[Path]) -> DatasetDict:
    train_sets = []
    val_sets = []
    test_sets = []
    for d in dirs:
        train_sets.append(Dataset.from_list(read_jsonl(d / "train.jsonl")))
        val_sets.append(Dataset.from_list(read_jsonl(d / "valid.jsonl")))
        test_sets.append(Dataset.from_list(read_jsonl(d / "test.jsonl")))
    dd = DatasetDict({
        "train": concatenate_datasets(train_sets) if len(train_sets) > 1 else train_sets[0],
        "validation": concatenate_datasets(val_sets) if len(val_sets) > 1 else val_sets[0],
        "test": concatenate_datasets(test_sets) if len(test_sets) > 1 else test_sets[0],
    })
    return dd

def compute_subtoken_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    # Inputs are already space-separated subtokens; if not, split
    ptoks = [t for t in pred.strip().split() if t]
    gtoks = [t for t in gold.strip().split() if t]
    if not ptoks and not gtoks:
        return 1.0, 1.0, 1.0
    if not ptoks or not gtoks:
        return 0.0, 0.0, 0.0
    pset = ptoks
    gset = gtoks
    # multiset overlap
    from collections import Counter
    pc = Counter(pset)
    gc = Counter(gset)
    inter = sum((pc & gc).values())
    prec = inter / sum(pc.values()) if pc else 0.0
    rec = inter / sum(gc.values()) if gc else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def group_by(dataset: Dataset, key: str) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(dataset):
        groups[r[key]].append(i)
    return groups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True, help="One or more dataset directories containing train.jsonl/valid.jsonl/test.jsonl")
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--model_name", default="Salesforce/codet5-small", type=str)
    ap.add_argument("--num_train_epochs", default=3, type=int)
    ap.add_argument("--per_device_train_batch_size", default=8, type=int)
    ap.add_argument("--per_device_eval_batch_size", default=8, type=int)
    ap.add_argument("--gradient_accumulation_steps", default=1, type=int)
    ap.add_argument("--learning_rate", default=3e-4, type=float)
    ap.add_argument("--weight_decay", default=0.0, type=float)
    ap.add_argument("--warmup_steps", default=1000, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--max_source_length", default=1024, type=int)
    ap.add_argument("--max_target_length", default=8, type=int)
    ap.add_argument("--eval_top_k", default=3, type=int, help="Compute top-k exact match using beam search.")
    ap.add_argument("--logging_steps", default=50, type=int)
    args = ap.parse_args()
    torch.cuda.set_device(1) 

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_dirs = [Path(d) for d in args.data_dirs]
    raw = load_multi_dataset_dirs(ds_dirs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    def preprocess(examples):
        inputs = examples["source"]
        targets = examples["target"]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            targets,
            max_length=args.max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        bf16=torch.cuda.is_available(),  # use bf16 if possible
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=[],
        seed=args.seed,
    )

    # We will compute metrics after training with a custom pass to get top-k and per-dataset.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Custom evaluation on test set with beam search for top-k and per-dataset breakdown
    model.eval()
    test_raw = raw["test"]
    test_tok = tokenized["test"]

    # Generate predictions
    gen_kwargs = dict(
        max_new_tokens=args.max_target_length + 2,
        num_beams=max(1, args.eval_top_k),
        num_return_sequences=max(1, args.eval_top_k),
        do_sample=False,
    )

    preds: List[List[str]] = []
    golds: List[str] = []
    datasets: List[str] = []
    langs: List[str] = []
    ids: List[str] = []

    test_loader = torch.utils.data.DataLoader(
        test_tok.with_format("torch"),
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    # We need original rows for meta fields
    original_rows = list(test_raw)
    assert len(original_rows) == len(test_tok)

    pred_idx = 0
    for batch in test_loader:
        batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs
            )
        # generated shape: (batch_size * num_return_sequences, seq_len)
        batch_size = batch["input_ids"].shape[0]
        num_ret = gen_kwargs["num_return_sequences"]
        # Decode
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        # Group per example
        for i in range(batch_size):
            start = i * num_ret
            ex_preds = [decoded[start + j].strip() for j in range(num_ret)]
            preds.append(ex_preds)
            # Attach gold and meta
            row = original_rows[pred_idx]
            golds.append(row["target"].strip())
            datasets.append(row["dataset"])
            langs.append(row["lang"])
            ids.append(row["id"])
            pred_idx += 1

    # Compute metrics
    overall = {
        "count": len(golds),
        "exact_match@1": 0,
        "topk_exact_match": 0,
        "topk": args.eval_top_k,
        "subtoken_precision": 0.0,
        "subtoken_recall": 0.0,
        "subtoken_f1": 0.0,
    }
    per_dataset: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "exact_match@1": 0,
        "topk_exact_match": 0,
        "topk": args.eval_top_k,
        "subtoken_precision": 0.0,
        "subtoken_recall": 0.0,
        "subtoken_f1": 0.0,
    })

    for i in range(len(golds)):
        gold = golds[i]
        cand_list = preds[i]
        first = cand_list[0] if cand_list else ""
        em1 = int(first.strip() == gold.strip())
        emk = int(gold.strip() in [c.strip() for c in cand_list])

        prec, rec, f1 = compute_subtoken_f1(first.lower(), gold.lower())

        overall["count"] += 0  # placeholder to keep keys
        overall["exact_match@1"] += em1
        overall["topk_exact_match"] += emk
        overall["subtoken_precision"] += prec
        overall["subtoken_recall"] += rec
        overall["subtoken_f1"] += f1

        ds = per_dataset[datasets[i]]
        ds["count"] += 1
        ds["exact_match@1"] += em1
        ds["topk_exact_match"] += emk
        ds["subtoken_precision"] += prec
        ds["subtoken_recall"] += rec
        ds["subtoken_f1"] += f1

    def finalize(stats: Dict[str, Any]):
        n = max(1, stats["count"])
        stats["exact_match@1"] = stats["exact_match@1"] / n
        stats["topk_exact_match"] = stats["topk_exact_match"] / n
        stats["subtoken_precision"] = stats["subtoken_precision"] / n
        stats["subtoken_recall"] = stats["subtoken_recall"] / n
        stats["subtoken_f1"] = stats["subtoken_f1"] / n

    finalize(overall)
    for ds in per_dataset.values():
        finalize(ds)

    results = {
        "overall": overall,
        "per_dataset": per_dataset,
        "config": vars(args),
        "num_examples": len(golds),
    }

    # Save predictions and metrics
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Write a Markdown table
    lines = []
    lines.append("# Results")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Exact Match@1 | {overall['exact_match@1']:.4f} |")
    lines.append(f"| Top-{overall['topk']} Exact Match | {overall['topk_exact_match']:.4f} |")
    lines.append(f"| Subtoken Precision | {overall['subtoken_precision']:.4f} |")
    lines.append(f"| Subtoken Recall | {overall['subtoken_recall']:.4f} |")
    lines.append(f"| Subtoken F1 | {overall['subtoken_f1']:.4f} |")
    lines.append("")
    lines.append("## Per-dataset")
    lines.append("")
    lines.append("| Dataset | Count | EM@1 | Top-k EM | P | R | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, st in sorted(per_dataset.items()):
        lines.append(f"| {name} | {st['count']} | {st['exact_match@1']:.4f} | {st['topk_exact_match']:.4f} | {st['subtoken_precision']:.4f} | {st['subtoken_recall']:.4f} | {st['subtoken_f1']:.4f} |")
    with (out_dir / "results.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also save raw predictions for inspection
    pred_rows = []
    for i in range(len(golds)):
        pred_rows.append({
            "id": ids[i],
            "dataset": datasets[i],
            "lang": langs[i],
            "target": golds[i],
            "predictions": preds[i],
        })
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in pred_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Evaluation complete. See results.md and results.json.")

if __name__ == "__main__":
    main()