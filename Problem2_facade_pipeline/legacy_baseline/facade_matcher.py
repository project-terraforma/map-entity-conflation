"""Baseline facade matching implementation."""
from shapely.ops import nearest_points

from config import ENTRANCE_PREFERRED_RADIUS_M
from confidence_scorer import score_match
from facade_extractor import extract_facades_from_building
from geometry_utils import point_to_point_bearing, safe_wkt


def nearest_layer_distance(layer_gdf, geometry):
    """Return distance to the nearest geometry in an optional layer."""
    if layer_gdf is None or layer_gdf.empty or geometry is None:
        return None
    try:
        return float(layer_gdf.geometry.distance(geometry).min())
    except Exception:
        return None


def choose_facade(poi_point, facades, entrance_gdf=None):
    """Choose a facade candidate, preferring nearest entrance when available."""
    candidates = []
    nearest_entrance = None
    nearest_entrance_to_poi_m = None
    if entrance_gdf is not None and not entrance_gdf.empty:
        try:
            entrance_distances = entrance_gdf.geometry.distance(poi_point)
            nearest_entrance_to_poi_m = float(entrance_distances.min())
            if nearest_entrance_to_poi_m <= ENTRANCE_PREFERRED_RADIUS_M:
                nearest_entrance = entrance_gdf.loc[entrance_distances.idxmin()].geometry
        except Exception:
            nearest_entrance = None

    for facade in facades:
        distance = float(poi_point.distance(facade["geometry"]))
        entrance_distance = None
        if nearest_entrance is not None:
            entrance_distance = float(nearest_entrance.distance(facade["geometry"]))
        candidates.append(
            {
                **facade,
                "distance_to_poi_m": distance,
                "distance_to_nearest_entrance_m": entrance_distance,
                "nearest_entrance_to_poi_m": nearest_entrance_to_poi_m,
            }
        )

    if not candidates:
        return None, None
    if nearest_entrance is not None:
        best = min(candidates, key=lambda c: (c["distance_to_nearest_entrance_m"], c["distance_to_poi_m"]))
        method = "entrance_supported_nearest_facade"
    else:
        best = min(candidates, key=lambda c: c["distance_to_poi_m"])
        method = "nearest_facade_edge"
    sorted_by_poi = sorted(candidates, key=lambda c: c["distance_to_poi_m"])
    second_best = sorted_by_poi[1]["distance_to_poi_m"] if len(sorted_by_poi) > 1 else None
    best["second_best_distance_m"] = second_best
    best["_candidate_facade_geometries"] = [candidate["geometry"] for candidate in candidates]
    return best, method


def no_match_record(poi_row, label, notes):
    """Return an output-shaped record for POIs that cannot be matched."""
    point = poi_row.geometry if "geometry" in poi_row.index else None
    return {
        "poi_id": poi_row.get("poi_id", ""),
        "poi_name": poi_row.get("poi_name", ""),
        "poi_category": poi_row.get("poi_category", ""),
        "poi_lon": point.x if point is not None else None,
        "poi_lat": point.y if point is not None else None,
        "matched_building_id": poi_row.get("matched_building_id", ""),
        "facade_id": "",
        "facade_index": None,
        "facade_wkt": "",
        "facade_midpoint_lon": None,
        "facade_midpoint_lat": None,
        "distance_to_facade_meters": None,
        "facade_bearing_degrees": None,
        "poi_to_facade_bearing_degrees": None,
        "method_used": "no_match",
        "confidence_score": 0.0,
        "confidence_label": label,
        "evidence_fields_used": "",
        "review_notes": notes,
    }


def match_poi_to_facade(poi_row, building_row, entrances_gdf=None, streets_gdf=None):
    """Match one projected POI point to one projected building footprint facade."""
    if building_row is None:
        return no_match_record(poi_row, "needs_review_no_building_match", "No matched building footprint was available.")

    poi_point = poi_row.geometry
    building_geom = building_row.geometry
    building_id = building_row.get("building_id", None)
    if not building_id:
        building_id = getattr(building_row, "name", "")
    if building_geom is None or building_geom.is_empty or not building_geom.is_valid:
        return no_match_record(poi_row, "needs_review_invalid_geometry", "Matched building geometry is missing or invalid.")

    facades = extract_facades_from_building(building_geom)
    if not facades:
        return no_match_record(poi_row, "needs_review_invalid_geometry", "No exterior facade edges could be extracted.")

    best, method = choose_facade(poi_point, facades, entrance_gdf=entrances_gdf)
    if best is None:
        return no_match_record(poi_row, "needs_review_invalid_geometry", "No facade candidate was available.")

    poi_inside = bool(building_geom.contains(poi_point) or building_geom.touches(poi_point))
    entrance_distance = best.get("distance_to_nearest_entrance_m")
    street_distance = nearest_layer_distance(streets_gdf, best["geometry"])
    score, label, evidence = score_match(
        best["distance_to_poi_m"],
        poi_inside_building=poi_inside,
        entrance_distance_m=entrance_distance,
        street_distance_m=street_distance,
        second_best_distance_m=best.get("second_best_distance_m"),
        has_building=True,
        invalid_geometry=False,
    )

    midpoint = best["midpoint"]
    nearest_on_facade = nearest_points(poi_point, best["geometry"])[1]
    return {
        "poi_id": poi_row.get("poi_id", ""),
        "poi_name": poi_row.get("poi_name", ""),
        "poi_category": poi_row.get("poi_category", ""),
        "poi_lon": poi_point.x,
        "poi_lat": poi_point.y,
        "matched_building_id": building_id,
        "building_match_method": poi_row.get("building_match_method", ""),
        "facade_id": f"{building_id}:{best['facade_index']}",
        "facade_index": best["facade_index"],
        "facade_wkt": safe_wkt(best["geometry"]),
        "facade_midpoint_lon": midpoint.x if midpoint is not None else None,
        "facade_midpoint_lat": midpoint.y if midpoint is not None else None,
        "distance_to_facade_meters": round(best["distance_to_poi_m"], 3),
        "second_best_facade_distance_m": round(best["second_best_distance_m"], 3) if best.get("second_best_distance_m") is not None else None,
        "facade_bearing_degrees": round(best["bearing_degrees"], 3) if best.get("bearing_degrees") is not None else None,
        "poi_to_facade_bearing_degrees": round(point_to_point_bearing(poi_point, nearest_on_facade), 3),
        "poi_inside_building": poi_inside,
        "nearest_entrance_to_poi_m": round(best.get("nearest_entrance_to_poi_m"), 3) if best.get("nearest_entrance_to_poi_m") is not None else None,
        "entrance_distance_to_facade_m": round(entrance_distance, 3) if entrance_distance is not None else None,
        "street_distance_to_facade_m": round(street_distance, 3) if street_distance is not None else None,
        "method_used": method,
        "confidence_score": score,
        "confidence_label": label,
        "evidence_fields_used": ";".join(evidence),
        "review_notes": "",
        "_selected_facade_geometry": best["geometry"],
        "_candidate_facade_geometries": best.get("_candidate_facade_geometries", []),
        "_building_geometry": building_geom,
        "_poi_geometry": poi_point,
    }


def run_matching(poi_gdf, buildings_gdf, entrances_gdf=None, streets_gdf=None):
    """Match every POI to a facade when a matched building footprint is available."""
    records = []
    if poi_gdf is None or buildings_gdf is None:
        print("POI or building GeoDataFrame missing; matching cannot run.")
        return records

    building_lookup = buildings_gdf.drop_duplicates("building_id").set_index("building_id")
    for _, poi in poi_gdf.iterrows():
        building_id = poi.get("matched_building_id", "")
        if not building_id or building_id not in building_lookup.index:
            records.append(no_match_record(poi, "needs_review_no_building_match", "POI has no usable building footprint match."))
            continue
        record = match_poi_to_facade(
            poi,
            building_lookup.loc[building_id],
            entrances_gdf=entrances_gdf,
            streets_gdf=streets_gdf,
        )
        records.append(record)
    return records
