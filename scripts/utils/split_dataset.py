import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

def load_jsonl(p: Path) -> List[Dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(rows: List[Dict], p: Path):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Split JSONL into train/val/test.")
    ap.add_argument("--input", required=True, type=str, help="Input raw.jsonl")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(Path(args.input))
    random.Random(args.seed).shuffle(rows)
    n = len(rows)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]

    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "valid.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print(f"Wrote to {out_dir}")

if __name__ == "__main__":
    main()