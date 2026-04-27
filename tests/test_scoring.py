from __future__ import annotations

import pandas as pd

from src.scoring import score_address_building, score_place_address


def test_address_building_scoring_rewards_containment() -> None:
    df = pd.DataFrame(
        {
            "point_within_polygon": [1, 0],
            "distance_point_to_polygon_m": [0.0, 10.0],
            "distance_point_to_centroid_m": [5.0, 12.0],
            "building_area_m2": [200.0, 200.0],
            "num_addresses_already_linked_to_building": [1, 1],
        }
    )
    weights = {
        "containment_bonus": 0.6,
        "polygon_distance_scale_m": 40.0,
        "centroid_distance_scale_m": 80.0,
        "building_area_cap_m2": 1500.0,
        "address_density_penalty": 0.08,
        "threshold": 0.45,
        "high_threshold": 0.8,
        "medium_threshold": 0.6,
    }
    scored = score_address_building(df, weights)
    assert scored.loc[0, "score"] > scored.loc[1, "score"]
    assert scored.loc[0, "confidence_tier"] in {"medium", "high"}


def test_place_address_scoring_rewards_same_building() -> None:
    df = pd.DataFrame(
        {
            "same_resolved_building": [1.0, 0.0],
            "distance_place_to_address_m": [5.0, 5.0],
            "street_name_match": [1.0, 1.0],
            "house_number_similarity_if_available": [1.0, 1.0],
            "place_name_support_score": [0.0, 0.0],
        }
    )
    weights = {
        "same_building_bonus": 0.45,
        "distance_scale_m": 45.0,
        "street_name_bonus": 0.2,
        "house_number_bonus": 0.15,
        "place_name_support_bonus": 0.1,
        "threshold": 0.4,
        "high_threshold": 0.75,
        "medium_threshold": 0.55,
    }
    scored = score_place_address(df, weights)
    assert scored.loc[0, "score"] > scored.loc[1, "score"]
