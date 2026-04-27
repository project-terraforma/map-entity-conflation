from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(data: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def first_existing_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def confidence_tier(score: pd.Series, high: float, medium: float, threshold: float) -> pd.Series:
    conditions = [
        score >= high,
        score >= medium,
        score >= threshold,
    ]
    return pd.Series(
        np.select(conditions, ["high", "medium", "low"], default="unresolved"),
        index=score.index,
    )


def bounded_inverse_distance(distance_m: pd.Series, scale_m: float) -> pd.Series:
    scale = max(scale_m, 1e-6)
    return 1.0 / (1.0 + distance_m.fillna(scale * 10.0) / scale)


def coalesce_series(df: pd.DataFrame, candidates: list[str], default: object = None) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return df[column]
    return pd.Series([default] * len(df), index=df.index)
