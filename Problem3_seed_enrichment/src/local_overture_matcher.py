"""Re-check imagery POI seeds against a local Overture place extract."""

import json
import math
from pathlib import Path

import pandas as pd

from config import (
    LOCAL_CONFIRMED_SCORE,
    LOCAL_CONFIRMED_NAME_SCORE,
    LOCAL_OVERTURE_PLACE_OPTIONS,
    LOCAL_POSSIBLE_SCORE,
    LOCAL_POSSIBLE_NAME_SCORE,
    LOCAL_SEARCH_RADIUS_M,
)
from text_utils import clean_text, first_nested_text, join_nested_text, similarity


def haversine_m(lat1, lon1, lat2, lon2):
    """Compute WGS84 point distance in meters."""
    radius_m = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def score_match(seed_name, candidate_name, distance_m, radius_m):
    """Score a candidate with name as the dominant signal."""
    name_score = similarity(seed_name, candidate_name)
    distance_score = max(0.0, 1.0 - float(distance_m) / radius_m)
    return round(0.75 * name_score + 0.25 * distance_score, 6), round(name_score, 6)


def _load_geojson(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") != "Point" or len(coordinates) < 2:
            continue
        rows.append({**properties, "poi_lon": coordinates[0], "poi_lat": coordinates[1]})
    return pd.DataFrame(rows)


def _standardize_places(df):
    rows = []
    for _, row in df.iterrows():
        geometry = row.get("geometry")
        lat = row.get("poi_lat")
        lon = row.get("poi_lon")
        if geometry is not None and hasattr(geometry, "geom_type") and geometry.geom_type == "Point":
            lon, lat = geometry.x, geometry.y
        if pd.isna(lat) or pd.isna(lon):
            continue
        rows.append(
            {
                "local_overture_id": clean_text(row.get("id") or row.get("poi_id")),
                "local_overture_name": first_nested_text(row.get("names") or row.get("name")),
                "local_overture_lat": float(lat),
                "local_overture_lon": float(lon),
                "local_overture_address": first_nested_text(row.get("addresses") or row.get("address")),
                "local_overture_category": first_nested_text(row.get("categories") or row.get("basic_category")),
                "local_overture_websites": join_nested_text(row.get("websites")),
                "local_overture_phones": join_nested_text(row.get("phones")),
                "local_overture_operating_status": clean_text(row.get("operating_status")),
            }
        )
    return pd.DataFrame(rows)


def load_local_places(explicit_paths=None):
    """Load and deduplicate available local Overture place extracts."""
    options = [Path(path) for path in explicit_paths] if explicit_paths else LOCAL_OVERTURE_PLACE_OPTIONS
    frames = []
    used_paths = []
    for path in options:
        if not path.exists():
            continue
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix.lower() in {".geojson", ".json"}:
            frame = _load_geojson(path)
        else:
            continue
        frames.append(_standardize_places(frame))
        used_paths.append(str(path.resolve()))
        if path.name in {"proxy_area_places.parquet", "places.geojson"}:
            break
    places = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not places.empty:
        places = places.drop_duplicates(subset=["local_overture_id"], keep="first")
    return places, used_paths


def _empty_match(status="not_found"):
    return {
        "local_overture_match_status": status,
        "local_overture_score": 0.0,
        "local_overture_name_score": 0.0,
        "local_overture_distance_m": "",
        "local_overture_id": "",
        "local_overture_name": "",
        "local_overture_address": "",
        "local_overture_category": "",
        "local_overture_websites": "",
        "local_overture_phones": "",
        "local_overture_operating_status": "",
    }


def match_one(seed, places, radius_m=LOCAL_SEARCH_RADIUS_M):
    """Return the best nearby local Overture candidate for one seed."""
    if places.empty:
        return _empty_match("local_extract_unavailable")
    candidates = []
    for _, place in places.iterrows():
        distance_m = haversine_m(seed["seed_lat"], seed["seed_lon"], place["local_overture_lat"], place["local_overture_lon"])
        if distance_m > radius_m:
            continue
        match_score, name_score = score_match(seed["poi_name_seed"], place["local_overture_name"], distance_m, radius_m)
        candidates.append({**place.to_dict(), "local_overture_score": match_score, "local_overture_name_score": name_score, "local_overture_distance_m": round(distance_m, 3)})
    if not candidates:
        return _empty_match()
    best = sorted(candidates, key=lambda row: (-row["local_overture_score"], row["local_overture_distance_m"]))[0]
    if best["local_overture_score"] >= LOCAL_CONFIRMED_SCORE and best["local_overture_name_score"] >= LOCAL_CONFIRMED_NAME_SCORE:
        best["local_overture_match_status"] = "confirmed"
    elif best["local_overture_score"] >= LOCAL_POSSIBLE_SCORE and best["local_overture_name_score"] >= LOCAL_POSSIBLE_NAME_SCORE:
        best["local_overture_match_status"] = "possible"
    else:
        return _empty_match()
    return best


def match_seeds(seeds, places):
    """Attach best local Overture match columns to every seed."""
    matches = [match_one(seed, places) for _, seed in seeds.iterrows()]
    return pd.concat([seeds.reset_index(drop=True), pd.DataFrame(matches)], axis=1)
