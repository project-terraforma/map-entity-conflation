"""Connect matched addresses to street names and optional street segments."""

import pandas as pd

from config import STREET_NEAREST_THRESHOLD_M
from normalization import extract_street_from_address, is_missing, normalize_street

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None


def add_address_derived_street(df):
    """Extract matched street from matched address text."""
    result = df.copy()
    result["matched_street"] = result["matched_address_text"].apply(extract_street_from_address)

    def status(row):
        if row.get("final_label") == "non_building_poi":
            return "not_applicable"
        if not is_missing(row.get("matched_street")):
            return "street_extracted_from_matched_address"
        return "no_street_candidate"

    result["street_connection_status"] = result.apply(status, axis=1)
    return result


def add_optional_street_segments(df, streets):
    """Find nearest optional Overture street segment and compare names."""
    result = df.copy()
    default_columns = {
        "nearest_street_segment_id": None,
        "nearest_street_name": None,
        "street_segment_distance_m": None,
        "street_segment_match_status": "not_available",
    }
    for column, value in default_columns.items():
        result[column] = value

    if streets is None or streets.empty or gpd is None:
        return result

    valid_streets = streets[streets["geometry"].apply(lambda g: g is not None and not g.is_empty)].copy()
    valid_points = result.dropna(subset=["matched_address_lat", "matched_address_lon"]).copy()
    if valid_streets.empty or valid_points.empty:
        return result

    street_gdf = gpd.GeoDataFrame(valid_streets, geometry="geometry", crs="EPSG:4326")
    point_gdf = gpd.GeoDataFrame(
        valid_points[["poi_id", "matched_address_lat", "matched_address_lon", "matched_street"]],
        geometry=gpd.points_from_xy(valid_points["matched_address_lon"], valid_points["matched_address_lat"]),
        crs="EPSG:4326",
    )
    projected_crs = point_gdf.estimate_utm_crs() or "EPSG:3857"
    nearest = gpd.sjoin_nearest(
        point_gdf.to_crs(projected_crs),
        street_gdf.to_crs(projected_crs),
        how="left",
        max_distance=STREET_NEAREST_THRESHOLD_M,
        distance_col="street_segment_distance_m",
    )
    nearest = nearest.dropna(subset=["street_segment_id"])
    if nearest.empty:
        return result

    nearest["street_segment_match_status"] = nearest.apply(
        lambda r: (
            "street_segment_name_match"
            if normalize_street(r.get("matched_street")) and normalize_street(r.get("matched_street")) == normalize_street(r.get("street_name"))
            else "street_segment_nearest_only"
        ),
        axis=1,
    )
    lookup = nearest.set_index("poi_id")[
        ["street_segment_id", "street_name", "street_segment_distance_m", "street_segment_match_status"]
    ]
    result = result.set_index("poi_id")
    for source, target in [
        ("street_segment_id", "nearest_street_segment_id"),
        ("street_name", "nearest_street_name"),
        ("street_segment_distance_m", "street_segment_distance_m"),
        ("street_segment_match_status", "street_segment_match_status"),
    ]:
        result.loc[lookup.index, target] = lookup[source]
    return result.reset_index()


def add_street_connection(df: pd.DataFrame, streets=None) -> pd.DataFrame:
    """Add address-derived and optional street-segment connection columns."""
    result = add_address_derived_street(df)
    return add_optional_street_segments(result, streets)
