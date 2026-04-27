from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import nearest_points

from .utils import normalize_text

LOGGER = logging.getLogger(__name__)


def _ensure_column(df: gpd.GeoDataFrame | pd.DataFrame, column: str, default: object) -> gpd.GeoDataFrame | pd.DataFrame:
    if column not in df.columns:
        df[column] = default
    return df


def _has_geometry(value: object) -> bool:
    return value is not None and hasattr(value, "geom_type")


def _prepare_addresses(addresses: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    addresses = addresses.copy()
    if "address_id" not in addresses.columns and "feature_id" in addresses.columns:
        addresses["address_id"] = addresses["feature_id"].astype(str)
    for column, default in [
        ("street_name", ""),
        ("street_name_norm", ""),
        ("house_number", ""),
        ("house_number_norm", ""),
    ]:
        addresses = _ensure_column(addresses, column, default)
    return addresses


def _prepare_buildings(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings = buildings.copy()
    if "building_id" not in buildings.columns and "feature_id" in buildings.columns:
        buildings["building_id"] = buildings["feature_id"].astype(str)
    buildings = _ensure_column(buildings, "street_name", "")
    buildings = _ensure_column(buildings, "street_name_norm", "")
    if "building_area_m2" not in buildings.columns:
        buildings["building_area_m2"] = buildings.geometry.area
    if "centroid_geometry" not in buildings.columns:
        buildings["centroid_geometry"] = buildings.geometry.centroid
    return buildings


def _prepare_places(places: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    places = places.copy()
    if "place_id" not in places.columns and "feature_id" in places.columns:
        places["place_id"] = places["feature_id"].astype(str)
    for column, default in [
        ("place_name", ""),
        ("street_name", ""),
        ("street_name_norm", ""),
        ("house_number", ""),
        ("house_number_norm", ""),
    ]:
        places = _ensure_column(places, column, default)
    return places


def _prepare_roads(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    roads = roads.copy()
    if "street_segment_id" not in roads.columns and "feature_id" in roads.columns:
        roads["street_segment_id"] = roads["feature_id"].astype(str)
    for column, default in [
        ("street_name", ""),
        ("street_name_norm", ""),
        ("road_class", "unknown"),
    ]:
        roads = _ensure_column(roads, column, default)
    return roads


def prepare_layer_keys(
    places: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    addresses: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Rename canonical feature ids into task-specific keys."""
    places = places.copy()
    buildings = buildings.copy()
    addresses = addresses.copy()
    roads = roads.copy()

    if not places.empty:
        places["place_id"] = places["feature_id"]
        places["place_name"] = places.get("feature_name", "")
    if not buildings.empty:
        buildings["building_id"] = buildings["feature_id"]
    if not addresses.empty:
        addresses["address_id"] = addresses["feature_id"]
    if not roads.empty:
        roads["road_id"] = roads["feature_id"]
        roads["street_segment_id"] = roads["feature_id"]
        roads["street_name"] = roads.get("street_name", roads.get("feature_name", ""))
        roads["street_name_norm"] = roads.get("street_name_norm", roads.get("feature_name_norm", ""))
    return places, buildings, addresses, roads


def compute_address_building_features(
    candidates: pd.DataFrame,
    addresses: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    addresses = _prepare_addresses(addresses)
    buildings = _prepare_buildings(buildings)
    merged = candidates.merge(
        addresses[["address_id", "geometry", "street_name", "street_name_norm", "house_number", "house_number_norm"]],
        on="address_id",
        how="left",
    ).rename(columns={"geometry": "address_geometry"})
    merged = merged.merge(
        buildings[["building_id", "geometry", "centroid_geometry", "building_area_m2", "street_name", "street_name_norm"]],
        on="building_id",
        how="left",
        suffixes=("", "_building"),
    ).rename(columns={"geometry": "building_geometry"})
    gdf = gpd.GeoDataFrame(merged, geometry="address_geometry", crs=addresses.crs)
    gdf["point_within_polygon"] = gdf.apply(
        lambda row: bool(row["building_geometry"].contains(row["address_geometry"])) if row["building_geometry"] else False,
        axis=1,
    )
    gdf["distance_point_to_polygon_m"] = gdf.apply(
        lambda row: row["address_geometry"].distance(row["building_geometry"]) if row["building_geometry"] else np.nan,
        axis=1,
    )
    gdf["distance_point_to_centroid_m"] = gdf.apply(
        lambda row: row["address_geometry"].distance(row["centroid_geometry"]) if row["centroid_geometry"] else np.nan,
        axis=1,
    )
    address_counts = gdf.groupby("building_id")["address_id"].nunique().rename("num_addresses_already_linked_to_building")
    gdf = gdf.merge(address_counts, on="building_id", how="left")
    gdf["match_reason"] = np.where(gdf["point_within_polygon"], "address inside building", "nearest building fallback")
    return pd.DataFrame(gdf)


def compute_place_building_features(
    candidates: pd.DataFrame,
    places: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    resolved_address_building: pd.DataFrame,
    addresses: gpd.GeoDataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    places = _prepare_places(places)
    buildings = _prepare_buildings(buildings)
    merged = candidates.merge(
        places[["place_id", "place_name", "geometry"]],
        on="place_id",
        how="left",
    ).rename(columns={"geometry": "place_geometry"})
    merged = merged.merge(
        buildings[["building_id", "geometry", "centroid_geometry", "building_area_m2"]],
        on="building_id",
        how="left",
    ).rename(columns={"geometry": "building_geometry"})
    gdf = gpd.GeoDataFrame(merged, geometry="place_geometry", crs=places.crs)
    gdf["point_within_polygon"] = gdf.apply(
        lambda row: bool(row["building_geometry"].contains(row["place_geometry"])) if row["building_geometry"] else False,
        axis=1,
    )
    gdf["distance_point_to_polygon_m"] = gdf.apply(
        lambda row: row["place_geometry"].distance(row["building_geometry"]) if row["building_geometry"] else np.nan,
        axis=1,
    )
    gdf["distance_point_to_centroid_m"] = gdf.apply(
        lambda row: row["place_geometry"].distance(row["centroid_geometry"]) if row["centroid_geometry"] else np.nan,
        axis=1,
    )
    near_counts = gdf.groupby("building_id")["place_id"].nunique().rename("num_places_near_building")
    gdf = gdf.merge(near_counts, on="building_id", how="left")

    if not resolved_address_building.empty and not addresses.empty:
        addresses = _prepare_addresses(addresses)
        linked_addresses = resolved_address_building.merge(
            addresses[["address_id", "geometry"]],
            on="address_id",
            how="left",
        )
        linked_addresses = linked_addresses.rename(columns={"geometry": "linked_address_geometry"})
        support = gdf.merge(linked_addresses, on="building_id", how="left")
        support["linked_distance"] = support.apply(
            lambda row: row["place_geometry"].distance(row["linked_address_geometry"])
            if _has_geometry(row.get("linked_address_geometry"))
            else np.nan,
            axis=1,
        )
        nearest_support = support.groupby(["place_id", "building_id"])["linked_distance"].min().rename("nearest_linked_address_distance_m")
        gdf = gdf.merge(nearest_support, on=["place_id", "building_id"], how="left")
    else:
        gdf["nearest_linked_address_distance_m"] = np.nan

    gdf["match_reason"] = np.where(gdf["point_within_polygon"], "place inside building", "nearest building fallback")
    return pd.DataFrame(gdf)


def compute_building_street_features(
    candidates: pd.DataFrame,
    buildings: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    resolved_address_building: pd.DataFrame,
    addresses: gpd.GeoDataFrame,
    frontage_buffer_m: float,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    buildings = _prepare_buildings(buildings)
    roads = _prepare_roads(roads)
    addresses = _prepare_addresses(addresses)
    merged = candidates.merge(
        buildings[["building_id", "geometry", "centroid_geometry", "street_name", "street_name_norm"]],
        on="building_id",
        how="left",
    ).rename(columns={"geometry": "building_geometry"})
    merged = merged.merge(
        roads[["street_segment_id", "street_name", "street_name_norm", "road_class", "geometry"]],
        on="street_segment_id",
        how="left",
        suffixes=("_building", "_road"),
    ).rename(columns={"geometry": "road_geometry"})
    gdf = gpd.GeoDataFrame(merged, geometry="building_geometry", crs=buildings.crs)
    gdf["min_distance_building_to_road_m"] = gdf.apply(
        lambda row: row["building_geometry"].distance(row["road_geometry"]) if row["road_geometry"] else np.nan,
        axis=1,
    )
    gdf["building_centroid_to_road_m"] = gdf.apply(
        lambda row: row["centroid_geometry"].distance(row["road_geometry"]) if row["road_geometry"] else np.nan,
        axis=1,
    )
    gdf["shared_frontage_proxy"] = gdf.apply(
        lambda row: float(row["building_geometry"].buffer(frontage_buffer_m).intersects(row["road_geometry"])) if row["road_geometry"] else 0.0,
        axis=1,
    )

    if not resolved_address_building.empty and not addresses.empty:
        linked_addresses = resolved_address_building.merge(
            addresses[["address_id", "building_id", "street_name_norm"]],
            on=["address_id", "building_id"],
            how="left",
        )
        dominant = linked_addresses.groupby("building_id")["street_name_norm"].agg(
            lambda values: next((value for value in values if value), "")
        )
        gdf = gdf.merge(dominant.rename("address_street_name_norm"), on="building_id", how="left")
    else:
        gdf["address_street_name_norm"] = ""

    gdf["road_name_matches_address_name"] = (
        gdf["street_name_norm_road"].fillna("") == gdf["address_street_name_norm"].fillna("")
    ).astype(float)
    gdf["is_corner_like_candidate"] = gdf.groupby("building_id")["street_segment_id"].transform("nunique").ge(2).astype(float)
    gdf["match_reason"] = np.where(gdf["shared_frontage_proxy"] > 0, "road intersects building frontage buffer", "nearest road fallback")
    return pd.DataFrame(gdf)


def compute_place_address_features(
    candidates: pd.DataFrame,
    places: gpd.GeoDataFrame,
    addresses: gpd.GeoDataFrame,
    resolved_place_building: pd.DataFrame,
    resolved_address_building: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    places = _prepare_places(places)
    addresses = _prepare_addresses(addresses)
    merged = candidates.merge(
        places[["place_id", "place_name", "geometry", "street_name", "street_name_norm", "house_number", "house_number_norm"]],
        on="place_id",
        how="left",
    ).rename(columns={"geometry": "place_geometry"})
    merged = merged.merge(
        addresses[["address_id", "geometry", "street_name", "street_name_norm", "house_number", "house_number_norm"]],
        on="address_id",
        how="left",
        suffixes=("_place", "_address"),
    ).rename(columns={"geometry": "address_geometry"})

    if not resolved_place_building.empty:
        merged = merged.merge(
            resolved_place_building[["place_id", "building_id"]].rename(columns={"building_id": "place_building_id"}),
            on="place_id",
            how="left",
        )
    else:
        merged["place_building_id"] = None
    if not resolved_address_building.empty:
        merged = merged.merge(
            resolved_address_building[["address_id", "building_id"]].rename(columns={"building_id": "address_building_id"}),
            on="address_id",
            how="left",
        )
    else:
        merged["address_building_id"] = None

    gdf = gpd.GeoDataFrame(merged, geometry="place_geometry", crs=places.crs)
    gdf["same_resolved_building"] = (gdf["place_building_id"] == gdf["address_building_id"]).fillna(False).astype(float)
    gdf["distance_place_to_address_m"] = gdf.apply(
        lambda row: row["place_geometry"].distance(row["address_geometry"])
        if _has_geometry(row.get("address_geometry"))
        else np.nan,
        axis=1,
    )
    gdf["street_name_match"] = (
        gdf["street_name_norm_place"].fillna("") == gdf["street_name_norm_address"].fillna("")
    ).astype(float)
    gdf["house_number_similarity_if_available"] = (
        gdf["house_number_norm_place"].fillna("").astype(str).str.strip().ne("")
        & (gdf["house_number_norm_place"].fillna("") == gdf["house_number_norm_address"].fillna(""))
    ).astype(float)
    gdf["shared_building_candidate"] = (gdf["candidate_source"] == "shared_building").astype(float)
    gdf["place_name_support_score"] = gdf.apply(
        lambda row: float(normalize_text(row["place_name"]) == normalize_text(row["street_name_address"]))
        if row.get("street_name_address") and normalize_text(row["place_name"])
        else 0.0,
        axis=1,
    )
    gdf["match_reason"] = np.where(gdf["same_resolved_building"] > 0, "place and address share resolved building", "nearest address fallback")
    return pd.DataFrame(gdf)


def estimate_entrance_points(building_street_resolved: pd.DataFrame, buildings: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Estimate a road-facing entrance point on the building boundary."""
    buildings = _prepare_buildings(buildings)
    roads = _prepare_roads(roads)
    if building_street_resolved.empty:
        return gpd.GeoDataFrame(columns=["building_id", "street_segment_id", "estimated_entrance_geometry"], geometry="estimated_entrance_geometry", crs=buildings.crs)

    merged = building_street_resolved.merge(
        buildings[["building_id", "geometry", "centroid_geometry"]],
        on="building_id",
        how="left",
    ).merge(
        roads[["street_segment_id", "geometry"]],
        on="street_segment_id",
        how="left",
        suffixes=("_building", "_road"),
    )
    points = []
    for _, row in merged.iterrows():
        building = row["geometry_building"]
        road = row["geometry_road"]
        if building is None or road is None:
            points.append(None)
            continue
        nearest_on_road, _ = nearest_points(road, building.centroid)
        _, nearest_on_building = nearest_points(building.boundary, nearest_on_road)
        points.append(nearest_on_building)

    output = merged[["building_id", "street_segment_id"]].copy()
    output["estimated_entrance_geometry"] = points
    return gpd.GeoDataFrame(output, geometry="estimated_entrance_geometry", crs=buildings.crs)
