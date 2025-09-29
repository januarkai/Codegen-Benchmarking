import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------- Data utils -----------------------------

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS, "<unk>": UNK}


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


_NON_ALNUM_RE = torch.compile if False else None  # placeholder to avoid import re at top
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
    vocab = dict(SPECIAL_TOKENS)
    for tok, _ in cnt.most_common(max(0, max_size - len(SPECIAL_TOKENS))):
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    unk = vocab.get("<unk>", UNK)
    return [vocab.get(t, unk) for t in tokens]


class NameDataset(Dataset):
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
        tgt_tokens = ["<bos>"] + [t for t in r["target"].split() if t][: self.max_tgt_len - 1] + ["<eos>"]
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
    pad_id = PAD
    max_src = max(len(x["input_ids"]) for x in batch)
    max_tgt = max(len(x["labels"]) for x in batch)
    input_ids = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for i, ex in enumerate(batch):
        input_ids[i, : len(ex["input_ids"])] = ex["input_ids"]
        labels[i, : len(ex["labels"]) ] = ex["labels"]
    meta = {
        "dataset": [x["dataset"] for x in batch],
        "lang": [x["lang"] for x in batch],
        "id": [x["id"] for x in batch],
        "target_str": [x["target_str"] for x in batch],
    }
    return {"input_ids": input_ids, "labels": labels, "meta": meta}


# ----------------------------- Model -----------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        x = x + self.pe[: x.size(0)]
        return x


class Code2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD)
        self.pos_enc = PositionalEncoding(d_model)
        self.pos_dec = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # src: (B, S) tgt: (B, T)
        src = src.t()  # (S, B)
        tgt = tgt.t()  # (T, B)
        src_key_padding_mask = src.eq(PAD).t()  # (B, S)
        tgt_key_padding_mask = tgt.eq(PAD).t()  # (B, T)
        src_emb = self.pos_enc(self.src_embed(src))
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        tgt_emb = self.pos_dec(self.tgt_embed(tgt))
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        dec = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        logits = self.out(dec)  # (T, B, vocab)
        return logits.transpose(0, 1)  # (B, T, vocab)

    @torch.no_grad()
    def beam_search(self, src: torch.Tensor, bos_id: int, eos_id: int, max_len: int, beam_size: int) -> List[List[int]]:
        # src: (B, S)
        device = src.device
        src_t = src.t()
        src_key_padding_mask = src_t.eq(PAD).t()
        memory = self.encoder(self.pos_enc(self.src_embed(src_t)), src_key_padding_mask=src_key_padding_mask)

        B = src.size(0)
        results: List[List[int]] = []
        for b in range(B):
            mem_b = memory[:, b:b+1, :]
            src_mask_b = src_key_padding_mask[b:b+1, :]
            beams = [(torch.tensor([bos_id], device=device, dtype=torch.long), 0.0)]
            completed: List[Tuple[torch.Tensor, float]] = []
            for _ in range(max_len):
                new_beams: List[Tuple[torch.Tensor, float]] = []
                for seq, score in beams:
                    if seq[-1].item() == eos_id:
                        completed.append((seq, score))
                        continue
                    tgt = seq.unsqueeze(1)  # (T,1)
                    tgt_emb = self.pos_dec(self.tgt_embed(tgt))
                    tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(device)
                    dec = self.decoder(tgt_emb, mem_b, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask_b)
                    logits = self.out(dec[-1, 0, :])  # (vocab,)
                    probs = torch.log_softmax(logits, dim=-1)
                    topk = torch.topk(probs, beam_size)
                    for idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
                        new_beams.append((torch.cat([seq, torch.tensor([idx], device=device)]), score + val))
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_size]
                if len(completed) >= beam_size:
                    break
            if not completed:
                completed = beams
            completed.sort(key=lambda x: x[1], reverse=True)
            bestk = [seq.tolist() for seq, _ in completed[:beam_size]]
            results.append(bestk)
        return results


# ----------------------------- Metrics -----------------------------


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


# ----------------------------- Main script -----------------------------


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ids_to_tokens(ids: List[int], vocab: Dict[str, int]) -> List[str]:
    inv = {v: k for k, v in vocab.items()}
    toks = []
    for i in ids:
        if i in (PAD, BOS):
            continue
        if i == EOS:
            break
        toks.append(inv.get(i, "<unk>"))
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--max_source_length", type=int, default=1024)
    ap.add_argument("--max_target_length", type=int, default=8)
    ap.add_argument("--eval_top_k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--src_vocab_size", type=int, default=50000)
    ap.add_argument("--tgt_vocab_size", type=int, default=5000)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1) 

    ds_dirs = [Path(d) for d in args.data_dirs]
    raw = load_multi_dataset_dirs(ds_dirs)

    # Build vocabularies on train set
    train_src_tokens = (tokenize_source(r["source"], args.max_source_length) for r in raw["train"])  # generator
    train_src_tokens = list(train_src_tokens)
    src_vocab = build_vocab(train_src_tokens, args.src_vocab_size)
    tgt_vocab = dict(SPECIAL_TOKENS)
    from collections import Counter
    cnt = Counter()
    for r in raw["train"]:
        cnt.update([t for t in r["target"].split() if t])
    for tok, _ in cnt.most_common(max(0, args.tgt_vocab_size - len(SPECIAL_TOKENS))):
        if tok not in tgt_vocab:
            tgt_vocab[tok] = len(tgt_vocab)

    # Datasets
    train_ds = NameDataset(raw["train"], src_vocab, tgt_vocab, args.max_source_length, args.max_target_length + 2)
    val_ds = NameDataset(raw["validation"], src_vocab, tgt_vocab, args.max_source_length, args.max_target_length + 2)
    test_ds = NameDataset(raw["test"], src_vocab, tgt_vocab, args.max_source_length, args.max_target_length + 2)

    train_loader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collate_fn)

    model = Code2SeqTransformer(len(src_vocab), len(tgt_vocab), d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    def run_epoch(loader: DataLoader, train: bool):
        model.train(train)
        total = 0.0
        steps = 0
        for batch in loader:
            inp = batch["input_ids"].to(device)
            lab = batch["labels"].to(device)
            # Teacher forcing: shift right inputs for decoder
            dec_in = lab[:, :-1]
            dec_out = lab[:, 1:]
            logits = model(inp, dec_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_out.reshape(-1))
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
        # simple checkpoint strategy
        if va < best_val:
            best_val = va
            torch.save({
                'model': model.state_dict(),
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'args': vars(args),
            }, out_dir / 'checkpoint.pt')

    # Load best
    ckpt = torch.load(out_dir / 'checkpoint.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    src_vocab = ckpt['src_vocab']
    tgt_vocab = ckpt['tgt_vocab']

    # Evaluation with beam search
    bos_id = tgt_vocab["<bos>"]
    eos_id = tgt_vocab["<eos>"]
    model.eval()
    preds_all: List[List[str]] = []
    golds_all: List[str] = []
    datasets: List[str] = []
    langs: List[str] = []
    ids: List[str] = []
    for batch in test_loader:
        inp = batch["input_ids"].to(device)
        beams = model.beam_search(inp, bos_id=bos_id, eos_id=eos_id, max_len=args.max_target_length + 2, beam_size=max(1, args.eval_top_k))
        # beams: List[B][K][toks]
        for i in range(len(beams)):
            seqs = []
            for s in beams[i]:
                toks = ids_to_tokens(s, tgt_vocab)
                seqs.append(" ".join(toks))
            preds_all.append(seqs)
        golds_all.extend(batch["meta"]["target_str"])
        datasets.extend(batch["meta"]["dataset"])
        langs.extend(batch["meta"]["lang"])
        ids.extend(batch["meta"]["id"])

    # Metrics
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

    results = {"overall": overall, "per_dataset": dict(per_dataset), "config": vars(args), "num_examples": len(golds_all), "algorithm": "code2seq"}

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown
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
