import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple

# Add the project root to Python's path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tqdm import tqdm

# Java parsing
try:
    import javalang
except Exception:
    javalang = None

# Python AST
import ast

from scripts.utils.subtokenize import subtokenize_identifier, normalize_target_tokens

MASK_TOKEN = "CLASSTOKEN"

def iter_files(root: Path, exts: Tuple[str, ...]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def mask_name_in_text(text: str, name: str) -> str:
    # Mask class name occurrences with word boundaries (case-sensitive)
    if not name:
        return text
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    return pattern.sub(MASK_TOKEN, text)

def extract_java_classes(file_path: Path) -> List[Dict]:
    if javalang is None:
        raise RuntimeError("javalang not installed. Please `pip install javalang`.")
    code = file_path.read_text(encoding="utf-8", errors="ignore")
    classes: List[Dict] = []
    try:
        tree = javalang.parse.parse(code)
    except Exception:
        return classes  # skip unparseable files
    # Collect classes via AST
    for path, node in tree.filter(javalang.tree.ClassDeclaration):
        class_name = node.name
        # Try to extract body text by matching braces after 'class <Name>'
        body_text = extract_java_class_body_text(code, class_name)
        if body_text is None:
            # fallback: whole file (but masked)
            body_text = code
        body_text = mask_name_in_text(body_text, class_name)
        classes.append({
            "name": class_name,
            "body": body_text,
        })
    return classes

def extract_java_class_body_text(code: str, class_name: str) -> Optional[str]:
    # Naive brace matching starting from "class <Name>"
    # Handles public final class, annotations, generics roughly.
    # Find the class keyword followed by the class name.
    # Make the search tolerant to modifiers in between lines.
    idx = None
    # Allow patterns like "class Name", "final class Name", "public class Name"
    # We'll simply search for "class\s+Name"
    m = re.search(rf"\bclass\s+{re.escape(class_name)}\b", code)
    if not m:
        return None
    start = m.end()
    # Find the first '{' after this match
    brace_start = code.find("{", start)
    if brace_start == -1:
        return None
    # Now match braces to find the end
    depth = 0
    i = brace_start
    while i < len(code):
        c = code[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                # body is between first '{' + 1 and this '}' - 1 inclusive
                return code[brace_start + 1:i]
        i += 1
    return None

def extract_python_classes(file_path: Path) -> List[Dict]:
    code = file_path.read_text(encoding="utf-8", errors="ignore")
    classes: List[Dict] = []
    try:
        tree = ast.parse(code)
    except Exception:
        return classes
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            # Use end_lineno if available to slice text
            try:
                start = node.lineno - 1
                end = getattr(node, "end_lineno", None)
                if end is None:
                    # Fallback: until EOF
                    end = len(code.splitlines())
                lines = code.splitlines()
                segment = "\n".join(lines[start:end])
            except Exception:
                segment = code
            segment = mask_name_in_text(segment, class_name)
            classes.append({
                "name": class_name,
                "body": segment,
            })
    return classes

def main():
    ap = argparse.ArgumentParser(description="Mine class-name dataset into JSONL.")
    ap.add_argument("--lang", choices=["java", "python"], required=True, help="Programming language.")
    ap.add_argument("--input", type=str, required=True, help="Path to file or directory containing sources.")
    ap.add_argument("--dataset", type=str, required=True, help="Dataset name to assign to records.")
    ap.add_argument("--output", type=str, required=True, help="Output JSONL file.")
    ap.add_argument("--id_prefix", type=str, default="", help="Optional prefix for IDs.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input path not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    if args.lang == "java":
        exts = (".java",)
        files = list(iter_files(in_path, exts)) if in_path.is_dir() else [in_path]
        for fp in tqdm(files, desc="Java files"):
            classes = extract_java_classes(fp)
            for idx, cls in enumerate(classes):
                target_tokens = subtokenize_identifier(cls["name"])
                if not target_tokens:
                    continue
                rec = {
                    "lang": "java",
                    "dataset": args.dataset,
                    "id": f"{args.id_prefix}{args.dataset}:{fp.as_posix()}:{cls['name']}:{idx}",
                    "source": cls["body"],
                    "target": " ".join(target_tokens),
                }
                records.append(rec)
    elif args.lang == "python":
        exts = (".py",)
        files = list(iter_files(in_path, exts)) if in_path.is_dir() else [in_path]
        for fp in tqdm(files, desc="Python files"):
            classes = extract_python_classes(fp)
            for idx, cls in enumerate(classes):
                target_tokens = subtokenize_identifier(cls["name"])
                if not target_tokens:
                    continue
                rec = {
                    "lang": "python",
                    "dataset": args.dataset,
                    "id": f"{args.id_prefix}{args.dataset}:{fp.as_posix()}:{cls['name']}:{idx}",
                    "source": cls["body"],
                    "target": " ".join(target_tokens),
                }
                records.append(rec)
    else:
        raise ValueError("Unsupported language.")

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")

if __name__ == "__main__":
    main()