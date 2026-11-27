# -*- coding: utf-8 -*-
"""Cryptographic SHA-256 hashing helpers for file integrity and row deduplication."""
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file using streaming reads."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_checksum_record(path: Path, base_dir: Path) -> dict:
    """Return relative path, byte size, and SHA-256 digest record."""
    p = Path(path)
    return {
        "path": str(p.relative_to(base_dir)).replace("\\", "/"),
        "size_bytes": p.stat().st_size,
        "sha256": sha256_file(p),
    }


def sha256_dict(payload: dict) -> str:
    """Deterministic SHA-256 of canonical JSON payload."""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def row_hash(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    """Deterministic 16-hex-char SHA-256 digest of normalized columns for deduplication."""
    joined = df[list(cols)].fillna("").astype(str).apply(lambda s: s.str.strip().str.upper()).agg("|".join, axis=1)
    return joined.map(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()[:16])
