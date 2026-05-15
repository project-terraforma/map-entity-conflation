"""Match POI and address points to building geometry."""

import pandas as pd

from config import BUILDING_NEAREST_FALLBACK_THRESHOLD_M
from normalization import is_missing

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None


def make_points(df, id_col, lat_col, lon_col):
    """Create a WGS84 point GeoDataFrame from lat/lon columns."""
    valid = df.dropna(subset=[lat_col, lon_col]).copy()
    if valid.empty or gpd is None:
        return None
    return gpd.GeoDataFrame(
        valid[[id_col, lat_col, lon_col]].copy(),
        geometry=gpd.points_from_xy(valid[lon_col], valid[lat_col]),
        crs="EPSG:4326",
    )


def make_building_gdf(buildings):
    """Create a GeoDataFrame of valid building geometry."""
    valid = buildings[buildings["geometry"].apply(lambda g: g is not None and not g.is_empty)].copy()
    if valid.empty or gpd is None:
        return None
    return gpd.GeoDataFrame(valid[["building_id", "geometry"]].copy(), geometry="geometry", crs="EPSG:4326")


def point_in_building(points, buildings, point_id_col):
    """Find containing buildings for point features."""
    if points is None or buildings is None or points.empty or buildings.empty:
        return {}
    joined = gpd.sjoin(points, buildings, how="left", predicate="within")
    joined = joined.dropna(subset=["building_id"])
    if joined.empty:
        return {}
    return joined.sort_values(point_id_col).groupby(point_id_col)["building_id"].first().to_dict()


def nearest_building(points, buildings, point_id_col, existing):
    """Find nearest building within configured fallback distance."""
    if points is None or buildings is None or points.empty or buildings.empty:
        return {}
    missing = points[~points[point_id_col].isin(existing.keys())].copy()
    if missing.empty:
        return {}
    projected_crs = points.estimate_utm_crs() or "EPSG:3857"
    point_proj = missing.to_crs(projected_crs)
    building_proj = buildings.to_crs(projected_crs)
    nearest = gpd.sjoin_nearest(
        point_proj,
        building_proj,
        how="left",
        max_distance=BUILDING_NEAREST_FALLBACK_THRESHOLD_M,
        distance_col="_building_distance_m",
    )
    nearest = nearest.dropna(subset=["building_id"])
    return nearest.groupby(point_id_col)["building_id"].first().to_dict()


def classify_building_status(row):
    """Classify how POI and matched address relate to buildings."""
    poi_b = row.get("poi_building_geom_id")
    addr_b = row.get("address_building_geom_id")
    if not is_missing(poi_b) and not is_missing(addr_b):
        return "building_consistent" if str(poi_b) == str(addr_b) else "building_conflict"
    if not is_missing(poi_b) or not is_missing(addr_b):
        return "building_possible"
    return "building_unknown"


def real_relation(row):
    """Return same/different/unknown building relation."""
    poi_b = row.get("poi_building_geom_id")
    addr_b = row.get("address_building_geom_id")
    if is_missing(poi_b) or is_missing(addr_b):
        return "building_unknown"
    return "same_building" if str(poi_b) == str(addr_b) else "different_building"


def match_buildings(df: pd.DataFrame, buildings: pd.DataFrame) -> pd.DataFrame:
    """Attach POI and matched-address building ids."""
    result = df.copy()
    result["poi_building_geom_id"] = result.get("overture_building_id", "").apply(
        lambda v: None if is_missing(v) else v
    )
    result["address_building_geom_id"] = None

    building_gdf = make_building_gdf(buildings)
    if gpd is not None and building_gdf is not None:
        poi_points = make_points(result, "poi_id", "poi_lat", "poi_lon")
        poi_inside = point_in_building(poi_points, building_gdf, "poi_id")
        poi_nearest = nearest_building(poi_points, building_gdf, "poi_id", poi_inside)
        poi_lookup = {**poi_nearest, **poi_inside}

        addr_points = make_points(
            result.dropna(subset=["matched_address_id"]).copy(),
            "poi_id",
            "matched_address_lat",
            "matched_address_lon",
        )
        addr_inside = point_in_building(addr_points, building_gdf, "poi_id")
        addr_nearest = nearest_building(addr_points, building_gdf, "poi_id", addr_inside)
        addr_lookup = {**addr_nearest, **addr_inside}

        result["poi_building_geom_id"] = result.apply(
            lambda r: r["poi_building_geom_id"] if not is_missing(r["poi_building_geom_id"]) else poi_lookup.get(r["poi_id"]),
            axis=1,
        )
        result["address_building_geom_id"] = result["poi_id"].map(addr_lookup)

    result["matched_building_id"] = result["poi_building_geom_id"].combine_first(result["address_building_geom_id"])
    result["building_status"] = result.apply(classify_building_status, axis=1)
    result["real_building_relation"] = result.apply(real_relation, axis=1)

    print(f"Building Matches: {result['matched_building_id'].notna().sum()}")
    return result
