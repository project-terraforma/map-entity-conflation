from __future__ import annotations

import random
from typing import Any

import pandas as pd

from .utils import write_json


def summarize_resolution_table(df: pd.DataFrame, name: str) -> dict[str, Any]:
    if df.empty:
        return {"table": name, "rows": 0, "resolved": 0, "unresolved": 0}
    resolved_mask = df["is_resolved"].fillna(False)
    return {
        "table": name,
        "rows": int(len(df)),
        "resolved": int(resolved_mask.sum()),
        "unresolved": int((~resolved_mask).sum()),
        "confidence_counts": df["confidence_tier"].value_counts(dropna=False).to_dict(),
        "score_summary": df["score"].describe().round(4).to_dict(),
    }


def compare_place_references(place_to_building: pd.DataFrame, reference_df: pd.DataFrame) -> dict[str, Any]:
    if place_to_building.empty or reference_df.empty or "overture_id" not in reference_df.columns:
        return {"reference_rows": 0, "matched_place_ids": 0}
    reference_ids = reference_df["overture_id"].dropna().astype(str).unique()
    matched = place_to_building["place_id"].astype(str).isin(reference_ids).sum()
    return {
        "reference_rows": int(len(reference_df)),
        "unique_reference_place_ids": int(len(reference_ids)),
        "matched_place_ids": int(matched),
    }


def build_error_samples(tables: dict[str, pd.DataFrame], sample_size: int = 25) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for table_name, df in tables.items():
        if df.empty:
            continue
        unresolved = df[~df["is_resolved"].fillna(False)].copy()
        if unresolved.empty:
            continue
        unresolved["error_bucket"] = unresolved.apply(_bucket_error, axis=1)
        unresolved["table_name"] = table_name
        if len(unresolved) > sample_size:
            unresolved = unresolved.sample(sample_size, random_state=42)
        rows.append(unresolved)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _bucket_error(row: pd.Series) -> str:
    reason = str(row.get("match_reason", "")).lower()
    if "corner" in reason:
        return "corner-lot street ambiguity"
    if "inside" in reason and row.get("score", 0.0) < 0.4:
        return "multiple plausible buildings"
    if "road" in reason:
        return "corner-lot street ambiguity"
    if "address" in reason:
        return "address far from all buildings"
    return "no nearby building"


def write_metrics_report(path: str, metrics: dict[str, Any]) -> None:
    write_json(metrics, path)
