from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import bounded_inverse_distance, confidence_tier


def _clip_score(score: pd.Series) -> pd.Series:
    return score.clip(lower=0.0, upper=1.0)


def score_address_building(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    scored = df.copy()
    score = (
        scored["point_within_polygon"].astype(float) * weights["containment_bonus"]
        + bounded_inverse_distance(scored["distance_point_to_polygon_m"], weights["polygon_distance_scale_m"]) * 0.25
        + bounded_inverse_distance(scored["distance_point_to_centroid_m"], weights["centroid_distance_scale_m"]) * 0.1
        + (scored["building_area_m2"].fillna(0.0).clip(upper=weights["building_area_cap_m2"]) / weights["building_area_cap_m2"]) * 0.05
        - scored["num_addresses_already_linked_to_building"].fillna(0.0) * weights["address_density_penalty"]
    )
    scored["score"] = _clip_score(score)
    scored["confidence_tier"] = confidence_tier(scored["score"], weights["high_threshold"], weights["medium_threshold"], weights["threshold"])
    return scored


def score_place_building(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    scored = df.copy()
    linked_distance = bounded_inverse_distance(
        scored["nearest_linked_address_distance_m"].fillna(weights["linked_address_distance_scale_m"] * 5.0),
        weights["linked_address_distance_scale_m"],
    )
    score = (
        scored["point_within_polygon"].astype(float) * weights["containment_bonus"]
        + bounded_inverse_distance(scored["distance_point_to_polygon_m"], weights["polygon_distance_scale_m"]) * 0.2
        + bounded_inverse_distance(scored["distance_point_to_centroid_m"], weights["centroid_distance_scale_m"]) * 0.1
        + linked_distance * 0.1
        + (scored["building_area_m2"].fillna(0.0).clip(upper=weights["building_area_cap_m2"]) / weights["building_area_cap_m2"]) * 0.05
        - (scored["num_places_near_building"].fillna(1.0) - 1.0).clip(lower=0.0) * weights["nearby_place_penalty"]
    )
    scored["score"] = _clip_score(score)
    scored["confidence_tier"] = confidence_tier(scored["score"], weights["high_threshold"], weights["medium_threshold"], weights["threshold"])
    return scored


def score_building_street(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    scored = df.copy()
    score = (
        bounded_inverse_distance(scored["min_distance_building_to_road_m"], weights["distance_scale_m"]) * 0.45
        + bounded_inverse_distance(scored["building_centroid_to_road_m"], weights["centroid_distance_scale_m"]) * 0.15
        + scored["shared_frontage_proxy"].fillna(0.0) * weights["frontage_bonus"]
        + scored["road_name_matches_address_name"].fillna(0.0) * weights["road_name_bonus"]
        + scored["is_corner_like_candidate"].fillna(0.0) * weights["corner_bonus"]
    )
    scored["score"] = _clip_score(score)
    scored["confidence_tier"] = confidence_tier(scored["score"], weights["high_threshold"], weights["medium_threshold"], weights["threshold"])
    return scored


def score_place_address(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    scored = df.copy()
    shared_building_candidate = (
        scored["shared_building_candidate"]
        if "shared_building_candidate" in scored.columns
        else pd.Series(0.0, index=scored.index)
    )
    score = (
        scored["same_resolved_building"].fillna(0.0) * weights["same_building_bonus"]
        + shared_building_candidate.fillna(0.0) * weights.get("shared_building_candidate_bonus", 0.1)
        + bounded_inverse_distance(scored["distance_place_to_address_m"], weights["distance_scale_m"]) * 0.35
        + scored["street_name_match"].fillna(0.0) * weights["street_name_bonus"]
        + scored["house_number_similarity_if_available"].fillna(0.0) * weights["house_number_bonus"]
        + scored["place_name_support_score"].fillna(0.0) * weights["place_name_support_bonus"]
    )
    scored["score"] = _clip_score(score)
    scored["confidence_tier"] = confidence_tier(scored["score"], weights["high_threshold"], weights["medium_threshold"], weights["threshold"])
    return scored
