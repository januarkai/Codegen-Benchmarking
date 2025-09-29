import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


PAD = 0
UNK = 1
SPECIAL = {"<pad>": PAD, "<unk>": UNK}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_multi_dataset_dirs(dirs: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    for d in dirs:
        train_rows.extend(read_jsonl(d / "train.jsonl"))
        val_rows.extend(read_jsonl(d / "valid.jsonl"))
        test_rows.extend(read_jsonl(d / "test.jsonl"))
    return {"train": train_rows, "validation": val_rows, "test": test_rows}


import re
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9_]+")


def tokenize_source(text: str, max_len: int) -> List[str]:
    toks = [t for t in _TOKEN_SPLIT.split(text) if t]
    return toks[:max_len]


def build_vocab(texts: Iterable[List[str]], max_size: int) -> Dict[str, int]:
    from collections import Counter
    cnt: Counter = Counter()
    for toks in texts:
        cnt.update(toks)
    vocab = dict(SPECIAL)
    for tok, _ in cnt.most_common(max(0, max_size - len(SPECIAL))):
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    unk = vocab.get("<unk>", UNK)
    return [vocab.get(t, unk) for t in tokens]


class BagDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], src_vocab: Dict[str, int], tgt_vocab: Dict[str, int], max_src_len: int, max_tgt_len: int):
        self.rows = rows
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        src_tokens = tokenize_source(r["source"], self.max_src_len)
        tgt_tokens = [t for t in r["target"].split() if t][: self.max_tgt_len]
        src_ids = encode(src_tokens, self.src_vocab)
        tgt_ids = encode(tgt_tokens, self.tgt_vocab)
        return {
            "input_ids": torch.tensor(src_ids, dtype=torch.long),
            "labels": torch.tensor(tgt_ids, dtype=torch.long),
            "dataset": r.get("dataset", "unknown"),
            "lang": r.get("lang", "unknown"),
            "id": r.get("id", str(idx)),
            "target_str": r["target"],
        }


def collate_fn(batch: List[Dict[str, Any]]):
    max_src = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((len(batch), max_src), PAD, dtype=torch.long)
    # Multi-label bag-of-subtokens target
    # Build a sparse multi-hot vector per example
    metas = {"dataset": [], "lang": [], "id": [], "target_str": []}
    labels: List[List[int]] = []
    for i, ex in enumerate(batch):
        input_ids[i, : len(ex["input_ids"])] = ex["input_ids"]
        metas["dataset"].append(ex["dataset"])
        metas["lang"].append(ex["lang"])
        metas["id"].append(ex["id"])
        metas["target_str"].append(ex["target_str"])
        labels.append(ex["labels"].tolist())
    return {"input_ids": input_ids, "labels_list": labels, "meta": metas}


class Code2VecLike(nn.Module):
    def __init__(self, vocab_size: int, tgt_vocab_size: int, hidden_dim: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD)
        self.encoder = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S)
        emb = self.embed(x)
        out, _ = self.encoder(emb)
        # mean pool
        pooled = out.mean(dim=1)
        logits = self.proj(pooled)
        return logits  # (B, tgt_vocab)


def compute_subtoken_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    ptoks = [t for t in pred.strip().split() if t]
    gtoks = [t for t in gold.strip().split() if t]
    if not ptoks and not gtoks:
        return 1.0, 1.0, 1.0
    if not ptoks or not gtoks:
        return 0.0, 0.0, 0.0
    from collections import Counter
    pc = Counter(ptoks)
    gc = Counter(gtoks)
    inter = sum((pc & gc).values())
    prec = inter / sum(pc.values()) if pc else 0.0
    rec = inter / sum(gc.values()) if gc else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=64)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--max_source_length", type=int, default=1024)
    ap.add_argument("--max_target_length", type=int, default=8)
    ap.add_argument("--eval_top_k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src_vocab_size", type=int, default=50000)
    ap.add_argument("--tgt_vocab_size", type=int, default=5000)
    ap.add_argument("--hidden_dim", type=int, default=512)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1) 

    ds_dirs = [Path(d) for d in args.data_dirs]
    raw = load_multi_dataset_dirs(ds_dirs)

    # Vocabs
    train_src_tokens = list(tokenize_source(r["source"], args.max_source_length) for r in raw["train"])  # type: ignore
    src_vocab = build_vocab(train_src_tokens, args.src_vocab_size)
    # Target vocab over subtokens
    from collections import Counter
    cnt = Counter()
    for r in raw["train"]:
        cnt.update([t for t in r["target"].split() if t])
    tgt_vocab = dict(SPECIAL)
    for tok, _ in cnt.most_common(max(0, args.tgt_vocab_size - len(SPECIAL))):
        if tok not in tgt_vocab:
            tgt_vocab[tok] = len(tgt_vocab)

    train_ds = BagDataset(raw["train"], src_vocab, tgt_vocab, args.max_source_length, args.max_target_length)
    val_ds = BagDataset(raw["validation"], src_vocab, tgt_vocab, args.max_source_length, args.max_target_length)
    test_ds = BagDataset(raw["test"], src_vocab, tgt_vocab, args.max_source_length, args.max_target_length)

    train_loader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collate_fn)

    model = Code2VecLike(len(src_vocab), len(tgt_vocab), hidden_dim=args.hidden_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # multi-label with sigmoid
    bce = nn.BCEWithLogitsLoss()

    def rows_to_multihot(rows: List[List[int]], vocab_size: int) -> torch.Tensor:
        y = torch.zeros((len(rows), vocab_size), dtype=torch.float32)
        for i, r in enumerate(rows):
            for t in r:
                if t < vocab_size:
                    y[i, t] = 1.0
        return y

    def run_epoch(loader: DataLoader, train: bool):
        model.train(train)
        total = 0.0
        steps = 0
        for batch in loader:
            x = batch["input_ids"].to(device)
            logits = model(x)
            y = rows_to_multihot(batch["labels_list"], logits.size(-1)).to(device)
            loss = bce(logits, y)
            if train:
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            total += loss.item()
            steps += 1
        return total / max(1, steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    for ep in range(1, args.num_train_epochs + 1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        if va < best_val:
            best_val = va
            torch.save({
                'model': model.state_dict(),
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'args': vars(args),
            }, out_dir / 'checkpoint.pt')

    ckpt = torch.load(out_dir / 'checkpoint.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    src_vocab = ckpt['src_vocab']
    tgt_vocab = ckpt['tgt_vocab']
    inv_tgt = {v: k for k, v in tgt_vocab.items()}

    # Evaluation: produce top-k subtokens independently (approximation)
    model.eval()
    preds_all: List[List[str]] = []
    golds_all: List[str] = []
    datasets: List[str] = []
    langs: List[str] = []
    ids: List[str] = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input_ids"].to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            topk = torch.topk(probs, k=max(1, args.eval_top_k), dim=-1).indices.cpu().tolist()
            for row in topk:
                # Interpret each candidate as a 1-token prediction; create k candidates.
                preds_all.append([inv_tgt.get(t, "") for t in row])
            golds_all.extend(batch["meta"]["target_str"])
            datasets.extend(batch["meta"]["dataset"])
            langs.extend(batch["meta"]["lang"])
            ids.extend(batch["meta"]["id"])

    def compute_subtoken_f1(pred: str, gold: str) -> Tuple[float, float, float]:
        ptoks = [t for t in pred.strip().split() if t]
        gtoks = [t for t in gold.strip().split() if t]
        if not ptoks and not gtoks:
            return 1.0, 1.0, 1.0
        if not ptoks or not gtoks:
            return 0.0, 0.0, 0.0
        from collections import Counter
        pc = Counter(ptoks)
        gc = Counter(gtoks)
        inter = sum((pc & gc).values())
        prec = inter / sum(pc.values()) if pc else 0.0
        rec = inter / sum(gc.values()) if gc else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    overall = {"count": len(golds_all), "exact_match@1": 0, "topk_exact_match": 0, "topk": args.eval_top_k, "subtoken_precision": 0.0, "subtoken_recall": 0.0, "subtoken_f1": 0.0}
    per_dataset: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "exact_match@1": 0, "topk_exact_match": 0, "topk": args.eval_top_k, "subtoken_precision": 0.0, "subtoken_recall": 0.0, "subtoken_f1": 0.0})
    for i, gold in enumerate(golds_all):
        cands = preds_all[i]
        first = cands[0] if cands else ""
        em1 = int(first.strip() == gold.strip())
        emk = int(gold.strip() in [c.strip() for c in cands[: args.eval_top_k]])
        p, r, f1 = compute_subtoken_f1(first.lower(), gold.lower())
        overall["exact_match@1"] += em1
        overall["topk_exact_match"] += emk
        overall["subtoken_precision"] += p
        overall["subtoken_recall"] += r
        overall["subtoken_f1"] += f1
        ds = per_dataset[datasets[i]]
        ds["count"] += 1
        ds["exact_match@1"] += em1
        ds["topk_exact_match"] += emk
        ds["subtoken_precision"] += p
        ds["subtoken_recall"] += r
        ds["subtoken_f1"] += f1

    def finalize(stats: Dict[str, Any]):
        n = stats.get("count", len(golds_all))
        if n <= 0:
            n = 1
        stats["exact_match@1"] = stats["exact_match@1"] / n
        stats["topk_exact_match"] = stats["topk_exact_match"] / n
        stats["subtoken_precision"] = stats["subtoken_precision"] / n
        stats["subtoken_recall"] = stats["subtoken_recall"] / n
        stats["subtoken_f1"] = stats["subtoken_f1"] / n

    finalize(overall)
    for st in per_dataset.values():
        finalize(st)

    results = {"overall": overall, "per_dataset": dict(per_dataset), "config": vars(args), "num_examples": len(golds_all), "algorithm": "code2vec"}

    out_dir = Path(args.output_dir)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

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

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for i in range(len(golds_all)):
            row = {"id": ids[i], "dataset": datasets[i], "lang": langs[i], "target": golds_all[i], "predictions": preds_all[i]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Evaluation complete. See results.md and results.json.")


if __name__ == "__main__":
    main()
