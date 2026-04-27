from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _empty_candidate_frame(left_id: str, right_id: str) -> pd.DataFrame:
    return pd.DataFrame(columns=[left_id, right_id, "candidate_source"])


def _combine_candidate_frames(frames: list[pd.DataFrame], left_id: str, right_id: str) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if not frame.empty]
    if not valid_frames:
        return _empty_candidate_frame(left_id, right_id)
    combined = pd.concat(valid_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=[left_id, right_id], keep="first")
    return combined.reset_index(drop=True)


def _spatial_join_contains(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    if points.empty or polygons.empty:
        return _empty_candidate_frame(left_label, right_label)
    joined = gpd.sjoin(
        points[[left_label, "geometry"]],
        polygons[[right_label, "geometry"]],
        how="inner",
        predicate="within",
    )
    return joined[[left_label, right_label]].assign(candidate_source="contains")


def _nearest_join(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    left_label: str,
    right_label: str,
    *,
    max_distance_m: float,
    nearest_n: int,
    exclusive_ids: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if left.empty or right.empty:
        return _empty_candidate_frame(left_label, right_label)

    subset = left.copy()
    if exclusive_ids is not None and not exclusive_ids.empty:
        subset = subset[~subset[left_label].isin(exclusive_ids[left_label])]
    if subset.empty:
        return _empty_candidate_frame(left_label, right_label)

    frames: list[pd.DataFrame] = []
    working = right[[right_label, "geometry"]].copy().reset_index(drop=True)
    spatial_index = working.sindex

    for left_row in subset[[left_label, "geometry"]].itertuples(index=False):
        geometry = left_row.geometry
        if geometry is None:
            continue
        search_geometry = geometry.buffer(max_distance_m)
        candidate_idx = list(spatial_index.query(search_geometry, predicate="intersects"))
        if not candidate_idx:
            continue
        candidates = working.iloc[candidate_idx].copy()
        candidates["candidate_distance_m"] = candidates.geometry.distance(geometry)
        candidates = candidates[candidates["candidate_distance_m"] <= max_distance_m]
        if candidates.empty:
            continue
        candidates = candidates.sort_values(["candidate_distance_m", right_label]).head(nearest_n).reset_index(drop=True)
        candidates[left_label] = getattr(left_row, left_label)
        candidates["candidate_source"] = [f"nearest_{rank}" for rank in range(1, len(candidates) + 1)]
        frames.append(candidates[[left_label, right_label, "candidate_distance_m", "candidate_source"]])

    return _combine_candidate_frames(frames, left_label, right_label)


def generate_address_building_candidates(
    addresses: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    *,
    max_distance_m: float,
    nearest_n: int,
) -> pd.DataFrame:
    contained = _spatial_join_contains(addresses, buildings, "address_id", "building_id")
    nearest = _nearest_join(
        addresses,
        buildings,
        "address_id",
        "building_id",
        max_distance_m=max_distance_m,
        nearest_n=nearest_n,
        exclusive_ids=contained[["address_id"]].drop_duplicates() if not contained.empty else None,
    )
    return _combine_candidate_frames([contained, nearest], "address_id", "building_id")


def generate_place_building_candidates(
    places: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    *,
    max_distance_m: float,
    nearest_n: int,
) -> pd.DataFrame:
    contained = _spatial_join_contains(places, buildings, "place_id", "building_id")
    nearest = _nearest_join(
        places,
        buildings,
        "place_id",
        "building_id",
        max_distance_m=max_distance_m,
        nearest_n=nearest_n,
        exclusive_ids=contained[["place_id"]].drop_duplicates() if not contained.empty else None,
    )
    return _combine_candidate_frames([contained, nearest], "place_id", "building_id")


def generate_building_street_candidates(
    buildings: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    *,
    max_distance_m: float,
    nearest_n: int,
) -> pd.DataFrame:
    if buildings.empty or roads.empty:
        return _empty_candidate_frame("building_id", "street_segment_id")

    working_buildings = buildings.copy()
    if "centroid_geometry" not in working_buildings.columns:
        working_buildings["centroid_geometry"] = working_buildings.geometry.centroid

    centroids = working_buildings[["building_id", "centroid_geometry"]].copy()
    centroids = centroids.rename(columns={"centroid_geometry": "geometry"})
    centroids = gpd.GeoDataFrame(centroids, geometry="geometry", crs=working_buildings.crs)

    nearest = _nearest_join(
        centroids,
        roads.rename(columns={"road_id": "street_segment_id"}),
        "building_id",
        "street_segment_id",
        max_distance_m=max_distance_m,
        nearest_n=nearest_n,
    )
    return nearest


def generate_place_address_candidates(
    places: gpd.GeoDataFrame,
    addresses: gpd.GeoDataFrame,
    place_building_resolved: pd.DataFrame,
    address_building_resolved: pd.DataFrame,
    *,
    max_distance_m: float,
    nearest_n: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if not place_building_resolved.empty and not address_building_resolved.empty:
        shared = place_building_resolved.merge(
            address_building_resolved[["address_id", "building_id"]],
            on="building_id",
            how="inner",
        )
        if not shared.empty:
            shared = shared[["place_id", "address_id"]].drop_duplicates()
            shared["candidate_source"] = "shared_building"
            frames.append(shared)

    if not places.empty and not addresses.empty:
        nearest = _nearest_join(
            places,
            addresses,
            "place_id",
            "address_id",
            max_distance_m=max_distance_m,
            nearest_n=nearest_n,
            exclusive_ids=frames[0][["place_id"]].drop_duplicates() if frames else None,
        )
        frames.append(nearest)

    return _combine_candidate_frames(frames, "place_id", "address_id")
