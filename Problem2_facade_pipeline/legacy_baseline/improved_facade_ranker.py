"""Prototype a lightweight facade re-ranker for existing Problem 2 matches.

This script does not change the production matching pipeline. It reconstructs
candidate facade edges from the matched building geometry, applies simple
rule-based ranking heuristics, and writes comparison metrics against the
existing facade match labels.
"""
from pathlib import Path

import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from confidence_scorer import score_match
from config import (
    BUILDINGS_INPUT_OPTIONS,
    MULTIPLE_CLOSE_FACADE_DELTA_M,
    OUTPUT_DIR,
    STREET_SUPPORT_RADIUS_M,
    STREETS_INPUT_OPTIONS,
)
from data_loader import find_input_file, load_table, parse_geometry
from facade_extractor import extract_facades_from_building
from geometry_utils import line_bearing, point_to_point_bearing, to_metric_crs


MATCHES_CSV = OUTPUT_DIR / "problem2_facade_matches.csv"
COMPARISON_CSV = OUTPUT_DIR / "problem2_ranker_comparison.csv"

CORNER_ENDPOINT_THRESHOLD_M = 4.0
SHORT_FACADE_LENGTH_M = 8.0
SAME_FACADE_POI_PENALTY_THRESHOLD = 5
ALIGNMENT_BONUS_M = 1.5
LONG_FACADE_BONUS_M = 1.0
CORNER_PENALTY_M = 1.25
SHORT_EDGE_PENALTY_M = 1.0
CROWDING_PENALTY_M = 0.75
RANK_SCORE_TIE_DELTA_M = 0.75


def note(message):
    print(f"NOTE: {message}")


def load_matches():
    if not MATCHES_CSV.exists():
        raise FileNotFoundError(f"Required match file not found: {MATCHES_CSV}")
    df = pd.read_csv(MATCHES_CSV, low_memory=False)
    print(f"Loaded matches: {MATCHES_CSV} ({len(df)} rows)")
    print(f"Match columns: {list(df.columns)}")
    return df


def load_buildings():
    if gpd is None:
        note("geopandas is unavailable; cannot reconstruct candidate facades.")
        return None
    path = find_input_file(BUILDINGS_INPUT_OPTIONS, "buildings", required=False)
    if path is None:
        note("building geometry file not found; cannot reconstruct candidate facades.")
        return None

    df = load_table(path)
    print(f"Loaded buildings: {path} ({len(df)} rows)")
    print(f"Building columns: {list(df.columns)}")
    if "id" not in df.columns or "geometry" not in df.columns:
        note("building file must contain existing columns 'id' and 'geometry'.")
        return None

    out = df[["id", "geometry"]].copy()
    out["geometry"] = out["geometry"].apply(parse_geometry)
    out = out[out["geometry"].apply(lambda geom: geom is not None and not geom.is_empty)].copy()
    if out.empty:
        note("no valid building geometries found.")
        return None
    return gpd.GeoDataFrame(out.rename(columns={"id": "building_id"}), geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326")


def load_streets(metric_crs):
    if gpd is None:
        return None
    path = find_input_file(STREETS_INPUT_OPTIONS, "streets", required=False)
    if path is None:
        note("street/segment geometry file not found; street alignment heuristic will be skipped.")
        return None
    df = load_table(path)
    print(f"Loaded streets: {path} ({len(df)} rows)")
    print(f"Street columns: {list(df.columns)}")
    if "geometry" not in df.columns:
        note("street file has no geometry column; street alignment heuristic will be skipped.")
        return None
    out = df[["geometry"]].copy()
    out["geometry"] = out["geometry"].apply(parse_geometry)
    out = out[out["geometry"].apply(lambda geom: geom is not None and not geom.is_empty and geom.geom_type in {"LineString", "MultiLineString"})].copy()
    if out.empty:
        note("no valid street line geometries found; street alignment heuristic will be skipped.")
        return None
    return gpd.GeoDataFrame(out, geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326").to_crs(metric_crs)


def angle_difference_degrees(a, b):
    if a is None or b is None:
        return None
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return min(diff, 180.0 - diff)


def nearest_street_info(streets_gdf, facade_geom):
    if streets_gdf is None or streets_gdf.empty or facade_geom is None:
        return None, None
    try:
        search_geom = facade_geom.buffer(STREET_SUPPORT_RADIUS_M)
        indices = list(streets_gdf.sindex.query(search_geom, predicate="intersects"))
        if indices:
            candidates = streets_gdf.iloc[indices]
        else:
            nearest = streets_gdf.sindex.nearest(facade_geom, return_all=False)
            if len(nearest) < 2 or len(nearest[1]) == 0:
                return None, None
            candidates = streets_gdf.iloc[list(nearest[1])]
        distances = candidates.geometry.distance(facade_geom)
        idx = distances.idxmin()
        nearest_geom = candidates.loc[idx].geometry
        return float(distances.loc[idx]), line_bearing(nearest_geom)
    except Exception:
        return None, None


def old_facade_crowding(matches):
    if "facade_id" not in matches.columns:
        note("facade_id missing; same-facade crowding penalty will be skipped.")
        return {}
    ids = matches["facade_id"].fillna("").astype(str)
    return ids[ids.ne("")].value_counts().to_dict()


def candidate_record(poi, building_id, facade, streets_gdf, crowding):
    poi_point = poi.geometry
    facade_geom = facade["geometry"]
    raw_distance = float(poi_point.distance(facade_geom))
    nearest_on_facade = nearest_points(poi_point, facade_geom)[1]
    coords = list(facade_geom.coords)
    endpoint_distance = min(nearest_on_facade.distance(Point(coords[0])), nearest_on_facade.distance(Point(coords[-1])))
    near_corner = endpoint_distance <= CORNER_ENDPOINT_THRESHOLD_M
    short_edge = facade["length_m"] <= SHORT_FACADE_LENGTH_M

    street_distance, street_bearing = nearest_street_info(streets_gdf, facade_geom)
    alignment_diff = angle_difference_degrees(facade.get("bearing_degrees"), street_bearing)
    street_aligned = (
        street_distance is not None
        and street_distance <= STREET_SUPPORT_RADIUS_M
        and alignment_diff is not None
        and alignment_diff <= 25.0
    )

    facade_id = f"{building_id}:{facade['facade_index']}"
    shared_poi_count = crowding.get(facade_id, 0)

    adjusted_score = raw_distance
    reasons = []
    if near_corner:
        adjusted_score += CORNER_PENALTY_M
        reasons.append("corner_endpoint_penalty")
    if short_edge:
        adjusted_score += SHORT_EDGE_PENALTY_M
        reasons.append("short_edge_penalty")
    if street_aligned:
        adjusted_score -= ALIGNMENT_BONUS_M
        reasons.append("street_alignment_bonus")
    if facade["length_m"] > SHORT_FACADE_LENGTH_M:
        adjusted_score -= min(LONG_FACADE_BONUS_M, facade["length_m"] / 50.0)
        reasons.append("long_facade_bonus")
    if shared_poi_count >= SAME_FACADE_POI_PENALTY_THRESHOLD:
        adjusted_score += CROWDING_PENALTY_M
        reasons.append("same_facade_crowding_penalty")

    return {
        "facade_id": facade_id,
        "facade_index": facade["facade_index"],
        "raw_distance_m": raw_distance,
        "adjusted_rank_score_m": adjusted_score,
        "facade_length_m": facade["length_m"],
        "near_corner": near_corner,
        "endpoint_distance_m": endpoint_distance,
        "street_distance_m": street_distance,
        "street_alignment_diff_degrees": alignment_diff,
        "street_aligned": street_aligned,
        "shared_poi_count": shared_poi_count,
        "reasons": reasons,
    }


def new_label_for_row(poi_row, building_geom, best, second_best):
    if best is None:
        return "needs_review_no_building_match", ""

    rank_gap = None
    if second_best is not None:
        rank_gap = second_best["adjusted_rank_score_m"] - best["adjusted_rank_score_m"]
    if rank_gap is not None and rank_gap <= RANK_SCORE_TIE_DELTA_M:
        return "needs_review_multiple_close_facades", "adjusted_rank_scores_still_close"

    poi_inside = bool(building_geom.contains(poi_row.geometry) or building_geom.touches(poi_row.geometry))
    score, label, evidence = score_match(
        best["raw_distance_m"],
        poi_inside_building=poi_inside,
        entrance_distance_m=None,
        street_distance_m=best["street_distance_m"],
        second_best_distance_m=None,
        has_building=True,
        invalid_geometry=False,
    )
    extra = []
    if rank_gap is not None:
        extra.append(f"adjusted_rank_gap_m={round(rank_gap, 3)}")
    extra.extend(best["reasons"])
    return label, ";".join(evidence + extra)


def rerank_matches(matches, buildings_m, streets_m):
    building_lookup = buildings_m.drop_duplicates("building_id").set_index("building_id")
    crowding = old_facade_crowding(matches)
    rows = []

    required = {"poi_lon", "poi_lat", "matched_building_id", "confidence_label"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Required columns missing from match output: {sorted(missing)}")

    for _, row in matches.iterrows():
        old_label = row.get("confidence_label", "")
        building_id = str(row.get("matched_building_id", "") or "")
        result = {
            "poi_id": row.get("poi_id", ""),
            "old_confidence_label": old_label,
            "new_confidence_label": old_label,
            "old_facade_id": row.get("facade_id", ""),
            "new_facade_id": row.get("facade_id", ""),
            "old_was_ambiguous": old_label == "needs_review_multiple_close_facades",
            "new_is_ambiguous": old_label == "needs_review_multiple_close_facades",
            "candidate_facade_count": pd.NA,
            "old_selected_rank": pd.NA,
            "new_rank_gap_m": pd.NA,
            "new_ranker_evidence": "",
        }

        if not building_id or building_id not in building_lookup.index:
            result["new_confidence_label"] = "needs_review_no_building_match"
            result["new_is_ambiguous"] = False
            result["new_ranker_evidence"] = "no_usable_building_match"
            rows.append(result)
            continue

        if pd.isna(row.get("poi_lon")) or pd.isna(row.get("poi_lat")):
            result["new_ranker_evidence"] = "missing_poi_coordinates"
            rows.append(result)
            continue

        poi_lon = pd.to_numeric(pd.Series([row.get("poi_lon")]), errors="coerce").iloc[0]
        poi_lat = pd.to_numeric(pd.Series([row.get("poi_lat")]), errors="coerce").iloc[0]
        if pd.isna(poi_lon) or pd.isna(poi_lat):
            result["new_ranker_evidence"] = "invalid_poi_coordinates"
            rows.append(result)
            continue

        poi_gdf = gpd.GeoDataFrame([{"geometry": Point(poi_lon, poi_lat)}], geometry="geometry", crs="EPSG:4326").to_crs(buildings_m.crs)
        poi_row = poi_gdf.iloc[0]
        building_geom = building_lookup.loc[building_id].geometry
        facades = extract_facades_from_building(building_geom)
        result["candidate_facade_count"] = len(facades)
        if not facades:
            result["new_confidence_label"] = "needs_review_invalid_geometry"
            result["new_is_ambiguous"] = False
            result["new_ranker_evidence"] = "no_facade_candidates"
            rows.append(result)
            continue

        candidates = [candidate_record(poi_row, building_id, facade, streets_m, crowding) for facade in facades]
        ranked = sorted(candidates, key=lambda item: item["adjusted_rank_score_m"])
        best = ranked[0]
        second_best = ranked[1] if len(ranked) > 1 else None

        old_facade_id = str(row.get("facade_id", "") or "")
        old_rank = next((idx + 1 for idx, candidate in enumerate(ranked) if candidate["facade_id"] == old_facade_id), pd.NA)
        result["old_selected_rank"] = old_rank
        result["new_facade_id"] = best["facade_id"]
        if second_best is not None:
            result["new_rank_gap_m"] = round(second_best["adjusted_rank_score_m"] - best["adjusted_rank_score_m"], 3)

        new_label, evidence = new_label_for_row(poi_row, building_geom, best, second_best)
        result["new_confidence_label"] = new_label
        result["new_is_ambiguous"] = new_label == "needs_review_multiple_close_facades"
        result["new_ranker_evidence"] = evidence
        rows.append(result)

    return pd.DataFrame(rows)


def count_table(series, prefix):
    counts = series.fillna("").replace("", "(blank)").value_counts()
    total = int(counts.sum())
    rows = []
    for label, count in counts.items():
        percent = round((int(count) / total) * 100, 2) if total else 0.0
        rows.append({"metric": prefix, "label": label, "count": int(count), "percent": percent})
    return rows


def write_comparison(detail):
    rows = []
    rows.extend(count_table(detail["old_confidence_label"], "old_confidence_label"))
    rows.extend(count_table(detail["new_confidence_label"], "new_confidence_label"))

    old_ambiguous = int(detail["old_was_ambiguous"].sum())
    new_ambiguous = int(detail["new_is_ambiguous"].sum())
    reduction = old_ambiguous - new_ambiguous
    reduction_percent = round((reduction / old_ambiguous) * 100, 2) if old_ambiguous else 0.0
    rows.append(
        {
            "metric": "ambiguity_reduction",
            "label": "needs_review_multiple_close_facades",
            "count": reduction,
            "percent": reduction_percent,
        }
    )

    changed = int((detail["old_facade_id"].fillna("").astype(str) != detail["new_facade_id"].fillna("").astype(str)).sum())
    rows.append({"metric": "facade_selection_changed", "label": "changed_facade_id", "count": changed, "percent": round((changed / len(detail)) * 100, 2)})

    comparison = pd.DataFrame(rows)
    comparison.to_csv(COMPARISON_CSV, index=False)
    print(f"Wrote comparison metrics: {COMPARISON_CSV}")
    return comparison


def print_findings(detail):
    old_ambiguous = int(detail["old_was_ambiguous"].sum())
    new_ambiguous = int(detail["new_is_ambiguous"].sum())
    reduction = old_ambiguous - new_ambiguous
    reduction_percent = round((reduction / old_ambiguous) * 100, 2) if old_ambiguous else 0.0
    changed = int((detail["old_facade_id"].fillna("").astype(str) != detail["new_facade_id"].fillna("").astype(str)).sum())

    print("\nConcise findings:")
    print(f"- Old ambiguous count: {old_ambiguous}")
    print(f"- New ambiguous count: {new_ambiguous}")
    print(f"- needs_review_multiple_close_facades reduction: {reduction} ({reduction_percent}%)")
    print(f"- Re-ranker changed selected facade IDs for {changed} POIs ({round((changed / len(detail)) * 100, 2)}%).")

    if "new_ranker_evidence" in detail.columns:
        evidence = detail["new_ranker_evidence"].fillna("").astype(str)
        for token in [
            "corner_endpoint_penalty",
            "short_edge_penalty",
            "street_alignment_bonus",
            "long_facade_bonus",
            "same_facade_crowding_penalty",
        ]:
            count = int(evidence.str.contains(token, regex=False).sum())
            print(f"- {token}: used in {count} selected new matches.")


def run():
    if gpd is None:
        raise ImportError("geopandas is required to reconstruct and rank facade candidates.")

    matches = load_matches()
    buildings = load_buildings()
    if buildings is None:
        raise RuntimeError("Cannot run improved ranker without existing building geometry.")

    buildings_m, metric_crs = to_metric_crs(buildings)
    streets_m = load_streets(metric_crs)
    detail = rerank_matches(matches, buildings_m, streets_m)
    comparison = write_comparison(detail)

    print("\nOld confidence label counts:")
    print(comparison[comparison["metric"] == "old_confidence_label"].to_string(index=False))
    print("\nNew confidence label counts:")
    print(comparison[comparison["metric"] == "new_confidence_label"].to_string(index=False))
    print_findings(detail)


if __name__ == "__main__":
    run()
