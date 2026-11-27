# -*- coding: utf-8 -*-
"""Shared JSON/CSV helpers and text/time normalization utilities."""
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def norm_text(s: Any) -> str:
    """Fold Unicode diacritics and lowercase alphanumeric tokens."""
    normalized = unicodedata.normalize("NFD", str(s or ""))
    ascii_folded = "".join(c for c in normalized if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^a-z0-9]+", "", ascii_folded.lower())


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
