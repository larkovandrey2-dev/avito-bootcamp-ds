from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_text(values: pd.Series) -> pd.Series:
    """Нормализация, общая для поиска и признаков."""
    return values.astype("string").fillna("").map(
        lambda value: re.sub(
            r"\s+", " ",
            re.sub(r"[^\w\s]+", " ", str(value).lower().replace("ё", "е")),
        ).strip()
    )


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).lower().replace("ё", "е").split())


def choose_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(path)


def remap_groups(groups: np.ndarray) -> np.ndarray:
    unique = np.unique(groups)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.asarray([mapping[value] for value in groups], dtype=np.int32)
