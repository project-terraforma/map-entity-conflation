from __future__ import annotations

import pandas as pd

from src.conflation_bundle import build_conflation_bundle


def test_conflation_bundle_marks_fully_resolved_consistent_rows() -> None:
    place_to_building = pd.DataFrame(
        {
            "place_id": ["p1"],
            "place_name": ["Cafe"],
            "building_id": ["b1"],
            "score": [0.9],
            "confidence_tier": ["high"],
            "match_reason": ["inside"],
            "is_resolved": [True],
        }
    )
    place_to_address = pd.DataFrame(
        {
            "place_id": ["p1"],
            "address_id": ["a1"],
            "score": [0.85],
            "confidence_tier": ["high"],
            "match_reason": ["shared building"],
            "is_resolved": [True],
        }
    )
    address_to_building = pd.DataFrame(
        {
            "address_id": ["a1"],
            "building_id": ["b1"],
            "score": [0.88],
            "confidence_tier": ["high"],
            "match_reason": ["inside"],
            "is_resolved": [True],
        }
    )
    building_to_street = pd.DataFrame(
        {
            "building_id": ["b1"],
            "street_segment_id": ["r1"],
            "street_name": ["Main Street"],
            "score": [0.8],
            "confidence_tier": ["high"],
            "match_reason": ["frontage"],
            "is_resolved": [True],
            "estimated_entrance_geometry": [None],
        }
    )

    bundle = build_conflation_bundle(place_to_building, place_to_address, address_to_building, building_to_street)
    assert bool(bundle.loc[0, "is_fully_resolved"]) is True
    assert bool(bundle.loc[0, "address_building_consistent"]) is True
    assert bundle.loc[0, "resolved_street_segment_id"] == "r1"
