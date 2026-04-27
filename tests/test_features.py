from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from src.feature_engineering import (
    compute_address_building_features,
    compute_building_street_features,
    prepare_layer_keys,
)


def test_address_building_features_detect_containment() -> None:
    addresses = gpd.GeoDataFrame({"feature_id": ["a1"]}, geometry=[Point(1, 1)], crs="EPSG:26913")
    buildings = gpd.GeoDataFrame({"feature_id": ["b1"]}, geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])], crs="EPSG:26913")
    _, buildings, addresses, _ = prepare_layer_keys(
        gpd.GeoDataFrame(geometry=[], crs="EPSG:26913"),
        buildings,
        addresses,
        gpd.GeoDataFrame(geometry=[], crs="EPSG:26913"),
    )
    candidates = pd.DataFrame({"address_id": ["a1"], "building_id": ["b1"], "candidate_source": ["contains"]})
    features = compute_address_building_features(candidates, addresses, buildings)
    assert bool(features.loc[0, "point_within_polygon"]) is True
    assert float(features.loc[0, "distance_point_to_polygon_m"]) == 0.0


def test_building_street_features_frontage_proxy() -> None:
    buildings = gpd.GeoDataFrame(
        {
            "feature_id": ["b1"],
            "street_name": ["Main Street"],
            "street_name_norm": ["main street"],
        },
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
        crs="EPSG:26913",
    )
    buildings["centroid_geometry"] = buildings.geometry.centroid
    addresses = gpd.GeoDataFrame(
        {"address_id": ["a1"], "building_id": ["b1"], "street_name_norm": ["main street"]},
        geometry=[Point(5, 5)],
        crs="EPSG:26913",
    )
    roads = gpd.GeoDataFrame(
        {
            "feature_id": ["r1"],
            "street_segment_id": ["r1"],
            "street_name": ["Main Street"],
            "street_name_norm": ["main street"],
            "road_class": ["residential"],
        },
        geometry=[LineString([(-5, -2), (15, -2)])],
        crs="EPSG:26913",
    )
    candidates = pd.DataFrame({"building_id": ["b1"], "street_segment_id": ["r1"], "candidate_source": ["nearest_1"]})
    resolved_ab = pd.DataFrame({"address_id": ["a1"], "building_id": ["b1"], "is_resolved": [True]})
    features = compute_building_street_features(candidates, buildings, roads, resolved_ab, addresses, frontage_buffer_m=3.0)
    assert float(features.loc[0, "shared_frontage_proxy"]) == 1.0
    assert float(features.loc[0, "road_name_matches_address_name"]) == 1.0
