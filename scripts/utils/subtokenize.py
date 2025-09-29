import re
from typing import List

_CAMEL_RE = re.compile(r"""
    # Split CamelCase words keeping sequences together:
    # e.g., HTTPServerError -> HTTP Server Error
    (?<=[a-z0-9])(?=[A-Z]) |       # aA
    (?<=[A-Z])(?=[A-Z][a-z])       # AAa
""", re.VERBOSE)

_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")

def split_camel(name: str) -> List[str]:
    # First split by non-alnum to handle snake_case and mixed cases
    parts = [p for p in _NON_ALNUM_RE.split(name) if p]
    out: List[str] = []
    for p in parts:
        out.extend(_CAMEL_RE.split(p))
    return out

def subtokenize_identifier(name: str) -> List[str]:
    # Split into subtokens and normalize to lowercase
    toks = split_camel(name)
    toks2: List[str] = []
    for t in toks:
        # Further split on digits boundaries if helpful, e.g., V2User -> ["V2", "User"]
        toks2.extend(re.findall(r"[A-Za-z]+|[0-9]+", t))
    return [t.lower() for t in toks2 if t]

def normalize_target_tokens(tokens: List[str]) -> str:
    return " ".join(tokens)