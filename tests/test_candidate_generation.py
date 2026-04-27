from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point

from src.candidate_generation import generate_place_address_candidates, generate_place_building_candidates


def test_place_building_candidates_return_multiple_nearest_matches() -> None:
    places = gpd.GeoDataFrame({"place_id": ["p1"]}, geometry=[Point(0, 0)], crs="EPSG:26913")
    buildings = gpd.GeoDataFrame(
        {"building_id": ["b1", "b2", "b3"]},
        geometry=[Point(1, 0).buffer(0.5), Point(2, 0).buffer(0.5), Point(3, 0).buffer(0.5)],
        crs="EPSG:26913",
    )

    candidates = generate_place_building_candidates(places, buildings, max_distance_m=10.0, nearest_n=3)
    assert candidates["building_id"].tolist() == ["b1", "b2", "b3"]
    assert candidates["candidate_source"].tolist() == ["nearest_1", "nearest_2", "nearest_3"]


def test_place_address_candidates_prefer_shared_building_but_keep_fallbacks() -> None:
    places = gpd.GeoDataFrame({"place_id": ["p1"]}, geometry=[Point(0, 0)], crs="EPSG:26913")
    addresses = gpd.GeoDataFrame(
        {"address_id": ["a1", "a2"]},
        geometry=[Point(0.5, 0), Point(2, 0)],
        crs="EPSG:26913",
    )
    place_building_resolved = gpd.GeoDataFrame({"place_id": ["p1"], "building_id": ["b1"]})
    address_building_resolved = gpd.GeoDataFrame({"address_id": ["a1"], "building_id": ["b1"]})

    candidates = generate_place_address_candidates(
        places,
        addresses,
        place_building_resolved,
        address_building_resolved,
        max_distance_m=10.0,
        nearest_n=2,
    )
    assert "shared_building" in set(candidates["candidate_source"])
    assert "nearest_1" not in set(candidates[candidates["address_id"] == "a1"]["candidate_source"])
