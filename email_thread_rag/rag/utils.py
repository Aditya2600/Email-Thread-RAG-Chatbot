from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
SUBJECT_PREFIX_RE = re.compile(r"^\s*((re|fw|fwd)\s*:\s*)+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def normalize_subject(subject: str) -> str:
    compact = SUBJECT_PREFIX_RE.sub("", subject or "").strip().lower()
    compact = re.sub(r"\s+", " ", compact)
    return compact


def overlap_chunks(tokens: list[str], chunk_size: int, overlap: int) -> Iterator[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    step = max(1, chunk_size - overlap)
    for start in range(0, len(tokens), step):
        yield tokens[start : start + chunk_size]
        if start + chunk_size >= len(tokens):
            break


def sliding_text_chunks(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    tokens = text.split()
    for token_window in overlap_chunks(tokens, chunk_size=chunk_size, overlap=overlap):
        yield " ".join(token_window)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True))
        handle.write("\n")


def coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if "," in value:
            return [part.strip() for part in value.split(",") if part.strip()]
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value).strip()]
