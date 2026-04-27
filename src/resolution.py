from __future__ import annotations

import pandas as pd


def resolve_best_matches(
    df: pd.DataFrame,
    group_key: str,
    score_col: str,
    threshold: float,
    min_score_gap: float = 0.1,
) -> pd.DataFrame:
    """Resolve one best match per group, unless the score is too weak or ambiguous."""
    if df.empty:
        return df.copy()

    ranked = df.sort_values([group_key, score_col], ascending=[True, False]).copy()
    ranked["rank"] = ranked.groupby(group_key)[score_col].rank(method="first", ascending=False)
    top = ranked[ranked["rank"] == 1].copy()
    second = ranked[ranked["rank"] == 2][[group_key, score_col]].rename(columns={score_col: "second_best_score"})
    top = top.merge(second, on=group_key, how="left")
    top["second_best_score"] = top["second_best_score"].fillna(0.0)
    top["score_gap"] = top[score_col] - top["second_best_score"]
    top["is_resolved"] = (top[score_col] >= threshold) & (top["score_gap"] >= min_score_gap)
    top.loc[~top["is_resolved"], "confidence_tier"] = "unresolved"
    return top.reset_index(drop=True)
