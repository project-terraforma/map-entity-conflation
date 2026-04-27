from __future__ import annotations

import pandas as pd

from src.resolution import resolve_best_matches


def test_resolution_requires_threshold_and_score_gap() -> None:
    df = pd.DataFrame(
        {
            "place_id": ["p1", "p1", "p2"],
            "building_id": ["b1", "b2", "b3"],
            "score": [0.9, 0.7, 0.41],
            "confidence_tier": ["high", "medium", "low"],
            "match_reason": ["x", "y", "z"],
        }
    )
    resolved = resolve_best_matches(df, "place_id", "score", threshold=0.45, min_score_gap=0.1)
    assert bool(resolved.loc[resolved["place_id"] == "p1", "is_resolved"].iloc[0]) is True
    assert bool(resolved.loc[resolved["place_id"] == "p2", "is_resolved"].iloc[0]) is False
    assert resolved.loc[resolved["place_id"] == "p2", "confidence_tier"].iloc[0] == "unresolved"
