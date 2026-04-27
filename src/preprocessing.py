from __future__ import annotations

import logging
from typing import Iterable

import geopandas as gpd
import pandas as pd

from .utils import first_existing_column, normalize_text

LOGGER = logging.getLogger(__name__)


def ensure_projected_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """Project a GeoDataFrame into a metric CRS for distance calculations."""
    if gdf.empty:
        return gdf.set_crs(target_crs, allow_override=True) if gdf.crs is None else gdf.to_crs(target_crs)
    if gdf.crs is None:
        LOGGER.warning("Input GeoDataFrame has no CRS; assuming EPSG:4326 before projection.")
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs) == str(target_crs):
        return gdf
    return gdf.to_crs(target_crs)


def _extract_first_non_null(row: pd.Series, candidates: Iterable[str]) -> object:
    for candidate in candidates:
        if candidate in row and pd.notna(row[candidate]):
            return row[candidate]
    return None


def canonicalize_layer(
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    *,
    id_candidates: list[str],
    name_candidates: list[str] | None = None,
    number_candidates: list[str] | None = None,
    street_candidates: list[str] | None = None,
    road_class_candidates: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Add normalized baseline columns so downstream code is schema-tolerant."""
    gdf = gdf.copy()
    id_col = first_existing_column(gdf.columns, id_candidates)
    if id_col is None:
        raise ValueError(f"{layer_name} layer is missing an identifier column from {id_candidates}.")
    gdf["feature_id"] = gdf[id_col].astype(str)

    if name_candidates:
        gdf["feature_name"] = gdf.apply(lambda row: _extract_first_non_null(row, name_candidates), axis=1)
        gdf["feature_name"] = gdf["feature_name"].fillna("")
        gdf["feature_name_norm"] = gdf["feature_name"].map(normalize_text)

    if number_candidates:
        gdf["house_number"] = gdf.apply(lambda row: _extract_first_non_null(row, number_candidates), axis=1)
        gdf["house_number_norm"] = gdf["house_number"].fillna("").astype(str).str.strip()

    if street_candidates:
        gdf["street_name"] = gdf.apply(lambda row: _extract_first_non_null(row, street_candidates), axis=1)
        gdf["street_name"] = gdf["street_name"].fillna("")
        gdf["street_name_norm"] = gdf["street_name"].map(normalize_text)

    if road_class_candidates:
        gdf["road_class"] = gdf.apply(lambda row: _extract_first_non_null(row, road_class_candidates), axis=1)
        gdf["road_class"] = gdf["road_class"].fillna("unknown").astype(str)

    if layer_name == "buildings" and gdf.crs is not None and not gdf.crs.is_geographic:
        gdf["building_area_m2"] = gdf.geometry.area
        gdf["centroid_geometry"] = gdf.geometry.centroid

    if "house_number_norm" not in gdf.columns:
        gdf["house_number_norm"] = ""

    return gdf
