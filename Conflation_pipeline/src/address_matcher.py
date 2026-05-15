"""POI to address matching with interpretable text and distance scoring."""

import math
from difflib import SequenceMatcher

import pandas as pd

from config import (
    ADDRESS_DISTANCE_THRESHOLD_M,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
)
from normalization import (
    extract_house_number,
    is_missing,
    normalize_street,
    normalize_text,
)

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:  # pragma: no cover
    gpd = None
    Point = None


def haversine_m(lat1, lon1, lat2, lon2):
    """Compute distance in meters between two WGS84 points."""
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return float("inf")
    radius_m = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def similarity(a, b):
    """Return normalized text similarity from 0 to 1."""
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def distance_score(distance_m, max_distance=ADDRESS_DISTANCE_THRESHOLD_M):
    """Convert distance into a 0..1 score."""
    if pd.isna(distance_m) or distance_m == float("inf"):
        return 0.0
    return max(0.0, 1.0 - min(float(distance_m), max_distance) / max_distance)


def score_candidate(poi_address, addr_text, distance_m):
    """Score a POI/address candidate using explainable features."""
    poi_street = normalize_street(poi_address)
    addr_street = normalize_street(addr_text)
    address_s = similarity(poi_address, addr_text)
    street_s = similarity(poi_street, addr_street)
    dist_s = distance_score(distance_m)

    poi_number = extract_house_number(poi_address)
    addr_number = extract_house_number(addr_text)
    number_score = 0.0
    if poi_number and addr_number:
        number_score = 1.0 if poi_number == addr_number else -0.5

    if is_missing(poi_address):
        confidence = 0.85 * dist_s
    else:
        confidence = 0.40 * dist_s + 0.25 * address_s + 0.15 * street_s + 0.20 * max(0.0, number_score)
        if number_score < 0:
            confidence -= 0.20

    return {
        "distance_score": round(float(dist_s), 6),
        "address_similarity": round(float(address_s), 6),
        "street_similarity": round(float(street_s), 6),
        "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
    }


def classify_match(confidence, candidate_count):
    """Classify address match quality."""
    if candidate_count == 0:
        return "no_candidate"
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "matched_high"
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "matched_medium"
    return "uncertain"


def empty_match_row(poi, candidate_count=0):
    """Build an output row when no address candidate is available."""
    return {
        "poi_id": poi.get("poi_id"),
        "poi_name": poi.get("poi_name"),
        "poi_lat": poi.get("poi_lat"),
        "poi_lon": poi.get("poi_lon"),
        "poi_address_input": poi.get("poi_address_input"),
        "poi_types": poi.get("poi_types"),
        "overture_building_id": poi.get("overture_building_id"),
        "matched_address_id": None,
        "matched_address_text": None,
        "matched_address_lat": None,
        "matched_address_lon": None,
        "distance_m": None,
        "distance_score": 0.0,
        "address_similarity": 0.0,
        "street_similarity": 0.0,
        "confidence": 0.0,
        "candidate_count": candidate_count,
        "address_match_status": "no_candidate",
    }


def build_spatial_candidates(pois, addresses):
    """Generate nearby POI/address candidates using GeoPandas when available."""
    valid_pois = pois.dropna(subset=["poi_lat", "poi_lon"]).copy()
    valid_addresses = addresses.dropna(subset=["address_lat", "address_lon"]).copy()
    if gpd is None or valid_pois.empty or valid_addresses.empty:
        return pd.DataFrame()

    poi_gdf = gpd.GeoDataFrame(
        valid_pois,
        geometry=gpd.points_from_xy(valid_pois["poi_lon"], valid_pois["poi_lat"]),
        crs="EPSG:4326",
    )
    addr_gdf = gpd.GeoDataFrame(
        valid_addresses,
        geometry=gpd.points_from_xy(valid_addresses["address_lon"], valid_addresses["address_lat"]),
        crs="EPSG:4326",
    )
    projected_crs = poi_gdf.estimate_utm_crs() or "EPSG:3857"
    poi_proj = poi_gdf.to_crs(projected_crs)
    addr_proj = addr_gdf.to_crs(projected_crs)

    buffers = poi_proj[["poi_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(ADDRESS_DISTANCE_THRESHOLD_M)
    joined = gpd.sjoin(addr_proj, buffers, how="inner", predicate="within")
    if joined.empty:
        return pd.DataFrame()

    poi_points = poi_proj.set_index("poi_id").geometry
    joined["distance_m"] = joined.apply(lambda r: r.geometry.distance(poi_points.loc[r["poi_id"]]), axis=1)
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))


def build_fallback_candidates(pois, addresses):
    """Generate nearby candidates without GeoPandas using a rough bbox prefilter."""
    rows = []
    valid_addresses = addresses.dropna(subset=["address_lat", "address_lon"]).copy()
    if valid_addresses.empty:
        return pd.DataFrame()

    for _, poi in pois.dropna(subset=["poi_lat", "poi_lon"]).iterrows():
        lat = float(poi["poi_lat"])
        lon = float(poi["poi_lon"])
        lat_delta = ADDRESS_DISTANCE_THRESHOLD_M / 111320.0
        lon_delta = ADDRESS_DISTANCE_THRESHOLD_M / max(1.0, 111320.0 * math.cos(math.radians(lat)))
        subset = valid_addresses[
            valid_addresses["address_lat"].between(lat - lat_delta, lat + lat_delta)
            & valid_addresses["address_lon"].between(lon - lon_delta, lon + lon_delta)
        ].copy()
        if subset.empty:
            continue
        subset["poi_id"] = poi["poi_id"]
        subset["distance_m"] = subset.apply(
            lambda r: haversine_m(lat, lon, r["address_lat"], r["address_lon"]),
            axis=1,
        )
        rows.append(subset[subset["distance_m"] <= ADDRESS_DISTANCE_THRESHOLD_M])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_text_only_candidates(pois, addresses):
    """Generate exact normalized address candidates when coordinates are missing."""
    poi_text = pois[pois["poi_address_input"].apply(lambda v: not is_missing(v))].copy()
    addr_text = addresses[addresses["address_text"].apply(lambda v: not is_missing(v))].copy()
    if poi_text.empty or addr_text.empty:
        return pd.DataFrame()
    poi_text["_norm_address"] = poi_text["poi_address_input"].apply(normalize_text)
    addr_text["_norm_address"] = addr_text["address_text"].apply(normalize_text)
    merged = poi_text.merge(addr_text, on="_norm_address", how="inner", suffixes=("", "_addr"))
    merged["distance_m"] = merged.apply(
        lambda r: haversine_m(r.get("poi_lat"), r.get("poi_lon"), r.get("address_lat"), r.get("address_lon")),
        axis=1,
    )
    return merged


def choose_best_candidates(pois, candidates):
    """Score candidates and select one best address per POI."""
    if candidates.empty:
        return pd.DataFrame([empty_match_row(row) for _, row in pois.iterrows()])

    scored_rows = []
    candidate_counts = candidates.groupby("poi_id").size().to_dict()
    for _, row in candidates.iterrows():
        scores = score_candidate(row.get("poi_address_input"), row.get("address_text"), row.get("distance_m"))
        scored_rows.append(
            {
                "poi_id": row.get("poi_id"),
                "matched_address_id": row.get("address_id"),
                "matched_address_text": row.get("address_text"),
                "matched_address_lat": row.get("address_lat"),
                "matched_address_lon": row.get("address_lon"),
                "distance_m": None if row.get("distance_m") == float("inf") else row.get("distance_m"),
                "candidate_count": candidate_counts.get(row.get("poi_id"), 0),
                **scores,
            }
        )
    scored = pd.DataFrame(scored_rows)
    scored = scored.sort_values(["poi_id", "confidence", "distance_m"], ascending=[True, False, True])
    best = scored.groupby("poi_id", as_index=False).first()
    best["address_match_status"] = best.apply(lambda r: classify_match(r["confidence"], r["candidate_count"]), axis=1)

    result = pois.merge(best, on="poi_id", how="left")
    missing_mask = result["matched_address_id"].isna()
    for column, value in {
        "distance_score": 0.0,
        "address_similarity": 0.0,
        "street_similarity": 0.0,
        "confidence": 0.0,
        "candidate_count": 0,
        "address_match_status": "no_candidate",
    }.items():
        result.loc[missing_mask, column] = value
    return result


def match_pois_to_addresses(pois: pd.DataFrame, addresses: pd.DataFrame) -> pd.DataFrame:
    """Match raw POIs to raw addresses."""
    spatial = build_spatial_candidates(pois, addresses)
    if spatial.empty and gpd is None:
        spatial = build_fallback_candidates(pois, addresses)
    text_only = build_text_only_candidates(pois, addresses)
    candidates = pd.concat([spatial, text_only], ignore_index=True, sort=False).drop_duplicates(
        subset=["poi_id", "address_id"], keep="first"
    )
    if not candidates.empty:
        poi_columns = [
            "poi_id",
            "poi_name",
            "poi_lat",
            "poi_lon",
            "poi_address_input",
            "poi_types",
            "overture_building_id",
        ]
        candidates = candidates.drop(
            columns=[col for col in poi_columns if col in candidates.columns and col != "poi_id"],
            errors="ignore",
        ).merge(pois[poi_columns], on="poi_id", how="left")
    result = choose_best_candidates(pois, candidates)

    found = result["matched_address_id"].notna().sum()
    no_candidates = (result["address_match_status"] == "no_candidate").sum()
    print(f"Address Matches Found: {found}")
    print(f"No Address Candidates: {no_candidates}")
    return result
