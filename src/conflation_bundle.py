from __future__ import annotations

import numpy as np
import pandas as pd


def _graph_confidence_tier(score: pd.Series, is_fully_resolved: pd.Series) -> pd.Series:
    tier = np.select(
        [score >= 0.8, score >= 0.6, score >= 0.4],
        ["high", "medium", "low"],
        default="unresolved",
    )
    tier = pd.Series(tier, index=score.index)
    tier.loc[~is_fully_resolved] = "partial"
    return tier


def build_conflation_bundle(
    place_to_building: pd.DataFrame,
    place_to_address: pd.DataFrame,
    address_to_building: pd.DataFrame,
    building_to_street: pd.DataFrame,
) -> pd.DataFrame:
    if place_to_building.empty and place_to_address.empty:
        return pd.DataFrame()

    bundle = place_to_building.rename(
        columns={
            "building_id": "resolved_building_id",
            "score": "place_building_score",
            "confidence_tier": "place_building_confidence_tier",
            "match_reason": "place_building_match_reason",
            "is_resolved": "place_building_is_resolved",
        }
    ).copy()

    if place_to_address.empty:
        bundle["address_id"] = None
        bundle["place_address_score"] = np.nan
        bundle["place_address_confidence_tier"] = "unresolved"
        bundle["place_address_match_reason"] = ""
        bundle["place_address_is_resolved"] = False
    else:
        place_address = place_to_address.rename(
            columns={
                "score": "place_address_score",
                "confidence_tier": "place_address_confidence_tier",
                "match_reason": "place_address_match_reason",
                "is_resolved": "place_address_is_resolved",
            }
        )
        bundle = bundle.merge(place_address, on="place_id", how="outer")

    address_building = address_to_building.rename(
        columns={
            "building_id": "address_building_id",
            "score": "address_building_score",
            "confidence_tier": "address_building_confidence_tier",
            "match_reason": "address_building_match_reason",
            "is_resolved": "address_building_is_resolved",
        }
    )
    bundle = bundle.merge(address_building, on="address_id", how="left")

    building_street = building_to_street.rename(
        columns={
            "building_id": "resolved_building_id",
            "street_segment_id": "resolved_street_segment_id",
            "street_name": "resolved_street_name",
            "score": "building_street_score",
            "confidence_tier": "building_street_confidence_tier",
            "match_reason": "building_street_match_reason",
            "is_resolved": "building_street_is_resolved",
        }
    )
    bundle = bundle.merge(building_street, on="resolved_building_id", how="left")

    bundle["address_building_consistent"] = (
        bundle["address_id"].isna()
        | bundle["address_building_id"].isna()
        | (bundle["resolved_building_id"] == bundle["address_building_id"])
    )
    bundle["is_fully_resolved"] = (
        bundle["place_building_is_resolved"].fillna(False)
        & bundle["place_address_is_resolved"].fillna(False)
        & bundle["address_building_is_resolved"].fillna(False)
        & bundle["building_street_is_resolved"].fillna(False)
        & bundle["address_building_consistent"].fillna(False)
    )

    score_columns = [
        "place_building_score",
        "place_address_score",
        "address_building_score",
        "building_street_score",
    ]
    bundle["overall_graph_score"] = bundle[score_columns].mean(axis=1, skipna=True)
    consistent_mask = bundle["address_building_consistent"].fillna(False)
    bundle.loc[consistent_mask, "overall_graph_score"] = (
        bundle.loc[consistent_mask, "overall_graph_score"] + 0.05
    ).clip(upper=1.0)
    bundle["graph_confidence_tier"] = _graph_confidence_tier(bundle["overall_graph_score"], bundle["is_fully_resolved"])
    return bundle
