"""Evaluate Problem 2 facade predictions against sign/entrance proxy points.

These metrics are proxy agreement metrics. The proxy facade edge is the matched
building edge closest to an observed sign/front-door/entrance point, not a
human-verified facade label.
"""

import math
import re

import pandas as pd
from shapely.geometry import LineString, Point

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover
    raise ImportError("Proxy facade evaluation requires geopandas.") from exc

from config import (
    MATCHES_OUTPUT,
    ACCEPT_ADJACENT_CORNER_FACADE_AS_NEAR_CORRECT,
    BUILDING_SIDE_DENSITY_WEIGHT,
    CORNER_AMBIGUITY_ANALYSIS_OUTPUT,
    CORNER_AMBIGUITY_DISTANCE_M,
    ENABLE_LOCAL_STOREFRONT_HEURISTICS,
    EXCLUDE_NON_BUILDING_POIS_FROM_PROXY_EVAL,
    FACADE_DENSITY_BUFFER_M,
    LOCAL_STOREFRONT_EVALUATION_OUTPUT,
    LOCAL_STOREFRONT_IMPROVED_ROWS_OUTPUT,
    LOCAL_STOREFRONT_SUMMARY_OUTPUT,
    LOCAL_STOREFRONT_WORSENED_ROWS_OUTPUT,
    NON_BUILDING_PROXY_FINAL_LABELS,
    NON_BUILDING_PROXY_KEYWORDS,
    PROBLEM1_OUTPUT,
    PROXY_BASELINE_CORRECT_RERANKER_WRONG_OUTPUT,
    PROXY_CANDIDATES_OUTPUT,
    PROXY_CONFIRMED_EVALUATED_ROWS_OUTPUT,
    PROXY_EVALUATED_BUILDING_BASED_ROWS_OUTPUT,
    PROXY_EVALUATION_OUTPUT,
    PROXY_BASELINE_CORRECT_RERANKER_WRONG_ROWS_OUTPUT,
    PROXY_MATCHED_ROWS_OUTPUT,
    PROXY_MATCHED_NOT_EVALUATED_OUTPUT,
    PROXY_NON_BUILDING_EXCLUDED_OUTPUT,
    PROXY_NEEDS_REVIEW_EVALUATED_ROWS_OUTPUT,
    PROXY_RERANKER_CORRECT_BASELINE_WRONG_OUTPUT,
    PROXY_RERANKER_IMPROVED_ROWS_OUTPUT,
    PROXY_NAME_MATCH_MAX_DISTANCE_M,
    PROXY_RANKER_COMPARISON_OUTPUT,
    PROXY_SUMMARY_OUTPUT,
    PROXY_UNMATCHED_ROWS_OUTPUT,
    POI_PROXY_CONSISTENCY_WEIGHT,
    SHARED_BUILDING_IMPROVED_ROWS_OUTPUT,
    SHARED_BUILDING_LOGIC_EVALUATION_OUTPUT,
    SHARED_BUILDING_LOGIC_SUMMARY_OUTPUT,
    SHARED_BUILDING_UNCHANGED_ROWS_OUTPUT,
    SHARED_BUILDING_WORSENED_ROWS_OUTPUT,
)
from proxy_benchmark_builder import count_raw_proxy_rows
from run_pipeline import distance_score, extract_facades, load_buildings


def normalize_name(value):
    """Normalize POI names for conservative fallback matching."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).lower()
    text = re.sub(r"\{.*?'primary':\s*'([^']+)'.*?\}", r"\1", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _clean_id(value):
    """Return a clean id string or empty string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _as_bool(value):
    """Parse common CSV boolean representations."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_required_csv(path, label):
    """Load a required CSV with a clear error."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path, low_memory=False)


def _load_inputs():
    """Load Problem 2 predictions, proxy points, and Problem 1 coordinates."""
    p2 = _load_required_csv(MATCHES_OUTPUT, "Problem 2 facade matches")
    proxy = _load_required_csv(PROXY_CANDIDATES_OUTPUT, "proxy candidates")
    p1 = _load_required_csv(PROBLEM1_OUTPUT, "Problem 1 output")
    return p2, proxy, p1


def _prepare_problem2_rows(p2, p1):
    """Attach Problem 1 coordinates/address fields to Problem 2 rows."""
    p1_cols = [
        col
        for col in [
            "poi_id",
            "poi_lat",
            "poi_lon",
            "matched_address_text",
            "matched_street",
            "building_validation_label",
            "poi_class",
            "poi_types",
            "final_label",
            "review_reason",
        ]
        if col in p1.columns
    ]
    merged = p2.merge(p1[p1_cols].drop_duplicates("poi_id"), on="poi_id", how="left", suffixes=("", "_problem1"))
    merged["_poi_id_clean"] = merged["poi_id"].apply(_clean_id)
    merged["_name_norm"] = merged["poi_name"].apply(normalize_name)
    return merged


def _distance_m_for_name_fallback(proxy_row, candidates_gdf):
    """Find nearest same-name Problem 2 row for a proxy point."""
    name = normalize_name(proxy_row.get("poi_name"))
    if not name:
        return None
    subset = candidates_gdf[candidates_gdf["_name_norm"] == name].copy()
    if subset.empty:
        return None
    point = gpd.GeoDataFrame(
        [{"geometry": Point(float(proxy_row["proxy_lon"]), float(proxy_row["proxy_lat"]))}],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_crs(candidates_gdf.crs).iloc[0].geometry
    distances = subset.geometry.distance(point)
    best_idx = distances.idxmin()
    best_distance = float(distances.loc[best_idx])
    if best_distance > PROXY_NAME_MATCH_MAX_DISTANCE_M:
        return None
    result = subset.loc[best_idx].copy()
    result["_match_distance_m"] = best_distance
    return result


def match_proxy_to_problem2(p2, proxy, p1):
    """Match proxy rows to one Problem 2 row using ids first, then name+nearby fallback."""
    p2_prepared = _prepare_problem2_rows(p2, p1)
    id_lookup = p2_prepared.drop_duplicates("_poi_id_clean").set_index("_poi_id_clean")

    p2_points = p2_prepared.dropna(subset=["poi_lat", "poi_lon"]).copy()
    p2_gdf = gpd.GeoDataFrame(
        p2_points,
        geometry=gpd.points_from_xy(p2_points["poi_lon"], p2_points["poi_lat"]),
        crs="EPSG:4326",
    )
    p2_gdf = p2_gdf.to_crs(p2_gdf.estimate_utm_crs() or "EPSG:3857") if not p2_gdf.empty else p2_gdf

    matched_rows = []
    unmatched_rows = []
    for _, proxy_row in proxy.iterrows():
        proxy_ids = [_clean_id(proxy_row.get("overture_poi_id")), _clean_id(proxy_row.get("poi_id"))]
        p2_row = None
        match_method = ""
        for proxy_id in proxy_ids:
            if proxy_id and proxy_id in id_lookup.index:
                p2_row = id_lookup.loc[proxy_id].copy()
                match_method = "overture_poi_id" if proxy_id == proxy_ids[0] else "poi_id"
                break
        if p2_row is None:
            p2_row = _distance_m_for_name_fallback(proxy_row, p2_gdf)
            if p2_row is not None:
                match_method = "normalized_name_nearby_coordinate"

        if p2_row is None:
            unmatched_rows.append({**proxy_row.to_dict(), "unmatched_reason": "no_unique_id_or_name_coordinate_match"})
            continue

        combined = {f"proxy_{key}": value for key, value in proxy_row.to_dict().items()}
        combined.update(p2_row.to_dict())
        combined["proxy_match_method"] = match_method
        matched_rows.append(combined)

    matched = pd.DataFrame(matched_rows)
    unmatched = pd.DataFrame(unmatched_rows)
    PROXY_MATCHED_ROWS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(PROXY_MATCHED_ROWS_OUTPUT, index=False)
    unmatched.to_csv(PROXY_UNMATCHED_ROWS_OUTPUT, index=False)
    return matched, unmatched


def _non_building_reason(row):
    """Return a reason when a matched proxy row is not eligible for facade evaluation."""
    final_label = str(row.get("source_final_label", row.get("final_label", row.get("problem1_final_label", ""))) or "")
    poi_class = str(row.get("poi_class", "") or "")
    if final_label in NON_BUILDING_PROXY_FINAL_LABELS:
        return f"final_label={final_label}"
    if poi_class == "non_building_poi":
        return "poi_class=non_building_poi"

    text = normalize_name(" ".join(str(row.get(col, "") or "") for col in ["poi_name", "poi_types", "proxy_notes"]))
    for keyword in NON_BUILDING_PROXY_KEYWORDS:
        if normalize_name(keyword) in text:
            return f"non_building_keyword={keyword}"
    return ""


def split_building_based_matches(matched):
    """Separate building-based matches from non-building rows before proxy accuracy."""
    if matched.empty or not EXCLUDE_NON_BUILDING_POIS_FROM_PROXY_EVAL:
        pd.DataFrame().to_csv(PROXY_NON_BUILDING_EXCLUDED_OUTPUT, index=False)
        return matched.copy(), pd.DataFrame()

    working = matched.copy()
    working["non_building_exclusion_reason"] = working.apply(_non_building_reason, axis=1)
    excluded = working[working["non_building_exclusion_reason"].astype(str) != ""].copy()
    eligible = working[working["non_building_exclusion_reason"].astype(str) == ""].copy()
    excluded.to_csv(PROXY_NON_BUILDING_EXCLUDED_OUTPUT, index=False)
    print(f"Wrote proxy non-building excluded rows: {PROXY_NON_BUILDING_EXCLUDED_OUTPUT} ({len(excluded)} rows)")
    return eligible, excluded


def line_bearing(line):
    """Return bearing for a LineString in projected coordinates."""
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    return (math.degrees(math.atan2(x2 - x1, y2 - y1)) + 360.0) % 360.0


def _facade_lookup(building_id, building_geom):
    """Build facade records keyed by edge id for one building."""
    facades = extract_facades(str(building_id), building_geom)
    return {facade["edge_id"]: facade for facade in facades}


def _parent_facade_id(edge_id):
    """Map optional split segment ids back to their parent facade id."""
    if ":seg" in str(edge_id):
        return str(edge_id).split(":seg", 1)[0]
    return edge_id


def _facade_index(edge_id):
    """Extract the numeric facade index from an edge id."""
    try:
        return int(str(edge_id).rsplit(":", 1)[1])
    except Exception:
        return None


def _facades_are_adjacent(edge_a, edge_b, facade_count):
    """Return True when two parent facade ids are adjacent on one exterior ring."""
    idx_a = _facade_index(edge_a)
    idx_b = _facade_index(edge_b)
    if idx_a is None or idx_b is None or facade_count <= 1:
        return False
    diff = abs(idx_a - idx_b)
    return diff == 1 or diff == facade_count - 1


def _evaluate_one(row, building_lookup, metric_crs):
    """Evaluate one matched proxy row."""
    if str(row.get("problem2_status", "")) != "facade_candidate" or not _clean_id(row.get("selected_facade_edge_id")):
        return None, "problem2_row_has_no_selected_facade"

    building_id = _clean_id(row.get("building_id"))
    if not building_id or building_id not in building_lookup.index:
        return None, "missing_building_geometry"

    proxy_lat = pd.to_numeric(pd.Series([row.get("proxy_proxy_lat")]), errors="coerce").iloc[0]
    proxy_lon = pd.to_numeric(pd.Series([row.get("proxy_proxy_lon")]), errors="coerce").iloc[0]
    if pd.isna(proxy_lat) or pd.isna(proxy_lon):
        return None, "missing_proxy_coordinates"

    proxy_point = gpd.GeoDataFrame(
        [{"geometry": Point(float(proxy_lon), float(proxy_lat))}],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_crs(metric_crs).iloc[0].geometry

    building_geom = building_lookup.loc[building_id].geometry
    facades = _facade_lookup(building_id, building_geom)
    if not facades:
        return None, "no_facade_edges"

    distances = {edge_id: proxy_point.distance(facade["geometry"]) for edge_id, facade in facades.items()}
    sorted_proxy_distances = sorted(distances.items(), key=lambda item: item[1])
    proxy_edge_id = sorted_proxy_distances[0][0]
    nearest_proxy_distance = float(sorted_proxy_distances[0][1])
    second_proxy_edge_id = sorted_proxy_distances[1][0] if len(sorted_proxy_distances) > 1 else None
    second_proxy_distance = float(sorted_proxy_distances[1][1]) if len(sorted_proxy_distances) > 1 else None
    proxy_gap = (second_proxy_distance - nearest_proxy_distance) if second_proxy_distance is not None else None
    selected_edge_id_raw = _clean_id(row.get("selected_facade_edge_id"))
    nearest_edge_id_raw = _clean_id(row.get("nearest_edge_id"))
    before_shared_edge_id_raw = _clean_id(row.get("selected_facade_before_shared_logic")) or selected_edge_id_raw
    after_shared_edge_id_raw = _clean_id(row.get("selected_facade_after_shared_logic")) or selected_edge_id_raw
    selected_edge_id = _parent_facade_id(selected_edge_id_raw)
    nearest_edge_id = _parent_facade_id(nearest_edge_id_raw)
    before_shared_edge_id = _parent_facade_id(before_shared_edge_id_raw)
    after_shared_edge_id = _parent_facade_id(after_shared_edge_id_raw)
    selected_facade = facades.get(selected_edge_id)
    nearest_facade = facades.get(nearest_edge_id)
    before_shared_facade = facades.get(before_shared_edge_id)
    after_shared_facade = facades.get(after_shared_edge_id)

    selected_distance = proxy_point.distance(selected_facade["geometry"]) if selected_facade else None
    nearest_distance = proxy_point.distance(nearest_facade["geometry"]) if nearest_facade else None
    before_shared_distance = proxy_point.distance(before_shared_facade["geometry"]) if before_shared_facade else None
    after_shared_distance = proxy_point.distance(after_shared_facade["geometry"]) if after_shared_facade else None
    proxy_distance = distances[proxy_edge_id]
    selected_agreement = bool(selected_edge_id and selected_edge_id == proxy_edge_id)
    nearest_agreement = bool(nearest_edge_id and nearest_edge_id == proxy_edge_id)
    before_shared_agreement = bool(before_shared_edge_id and before_shared_edge_id == proxy_edge_id)
    after_shared_agreement = bool(after_shared_edge_id and after_shared_edge_id == proxy_edge_id)
    selected_adjacent = _facades_are_adjacent(selected_edge_id, proxy_edge_id, len(facades))
    near_correct = bool(selected_agreement or (ACCEPT_ADJACENT_CORNER_FACADE_AS_NEAR_CORRECT and selected_adjacent and proxy_gap is not None and proxy_gap <= CORNER_AMBIGUITY_DISTANCE_M))

    result = {
        "poi_id": row.get("poi_id"),
        "poi_name": row.get("poi_name"),
        "building_id": building_id,
        "poi_lat": row.get("poi_lat"),
        "poi_lon": row.get("poi_lon"),
        "proxy_lat": proxy_lat,
        "proxy_lon": proxy_lon,
        "source_dataset": row.get("proxy_source_dataset"),
        "proxy_point_type": row.get("proxy_proxy_point_type"),
        "source_final_label": row.get("source_final_label", row.get("final_label", row.get("problem1_final_label"))),
        "problem2_eval_mode": row.get("problem2_eval_mode", "strict"),
        "proxy_eval_included_needs_review": _as_bool(row.get("proxy_eval_included_needs_review")),
        "selected_facade_edge_id": selected_edge_id,
        "nearest_edge_id": nearest_edge_id,
        "selected_facade_edge_id_raw": selected_edge_id_raw,
        "nearest_edge_id_raw": nearest_edge_id_raw,
        "proxy_facade_edge_id": proxy_edge_id,
        "proxy_agreement": selected_agreement,
        "nearest_edge_proxy_agreement": nearest_agreement,
        "selected_facade_before_shared_logic": before_shared_edge_id,
        "selected_facade_after_shared_logic": after_shared_edge_id,
        "selected_facade_before_shared_logic_raw": before_shared_edge_id_raw,
        "selected_facade_after_shared_logic_raw": after_shared_edge_id_raw,
        "shared_logic_changed_selection": bool(before_shared_edge_id != after_shared_edge_id),
        "shared_logic_improved_proxy_match": bool(after_shared_agreement and not before_shared_agreement),
        "shared_logic_worsened_proxy_match": bool(before_shared_agreement and not after_shared_agreement),
        "proxy_distance_before_shared_logic": round(float(before_shared_distance), 3) if before_shared_distance is not None else None,
        "proxy_distance_after_shared_logic": round(float(after_shared_distance), 3) if after_shared_distance is not None else None,
        "before_shared_logic_proxy_agreement": before_shared_agreement,
        "after_shared_logic_proxy_agreement": after_shared_agreement,
        "shared_building_poi_count": row.get("shared_building_poi_count"),
        "is_shared_building": _as_bool(row.get("is_shared_building")),
        "possible_mall_or_plaza": _as_bool(row.get("possible_mall_or_plaza")),
        "shared_building_mode_applied": _as_bool(row.get("shared_building_mode_applied")),
        "selected_facade_distance_to_proxy_point_m": round(float(selected_distance), 3) if selected_distance is not None else None,
        "nearest_edge_distance_to_proxy_point_m": round(float(nearest_distance), 3) if nearest_distance is not None else None,
        "proxy_point_distance_to_proxy_edge_m": round(float(proxy_distance), 3),
        "nearest_proxy_facade_distance_m": round(nearest_proxy_distance, 3),
        "second_nearest_proxy_facade_distance_m": round(second_proxy_distance, 3) if second_proxy_distance is not None else None,
        "proxy_facade_distance_gap_m": round(float(proxy_gap), 3) if proxy_gap is not None else None,
        "is_corner_ambiguous": bool(proxy_gap is not None and proxy_gap <= CORNER_AMBIGUITY_DISTANCE_M),
        "predicted_is_adjacent_to_proxy_facade": bool(selected_adjacent),
        "near_correct_under_corner_rule": near_correct,
        "selected_method": row.get("selected_method"),
        "problem2_status": row.get("problem2_status"),
        "selected_facade_score": row.get("selected_facade_score"),
        "street_used_for_rerank": row.get("street_used_for_rerank"),
        "proxy_match_method": row.get("proxy_match_method"),
        "notes": row.get("proxy_notes"),
    }
    return result, None


def _agreement_rate(df, group_col):
    """Return summary rows for agreement by group."""
    rows = []
    if df.empty or group_col not in df.columns:
        return rows
    grouped = df.groupby(group_col, dropna=False)
    for label, group in grouped:
        total = len(group)
        agree = int(group["proxy_agreement"].sum())
        rows.append(
            {
                "metric": f"agreement_rate_by_{group_col}",
                "group": str(label),
                "count": total,
                "agreement_count": agree,
                "value": round(agree / total, 6) if total else 0.0,
            }
        )
    return rows


def write_summary(
    evaluated,
    proxy_total,
    usable_total,
    matched_total,
    unmatched_total,
    not_evaluated_matched_total,
    non_building_excluded_total=0,
    building_based_matched_total=None,
):
    """Write proxy summary CSV."""
    if building_based_matched_total is None:
        building_based_matched_total = matched_total
    evaluated_count = len(evaluated)
    agreement_count = int(evaluated["proxy_agreement"].sum()) if evaluated_count else 0
    baseline_agreement_count = int(evaluated["nearest_edge_proxy_agreement"].sum()) if evaluated_count else 0
    if evaluated_count and "proxy_eval_included_needs_review" in evaluated.columns:
        needs_review_mask = evaluated["proxy_eval_included_needs_review"].apply(_as_bool)
    elif evaluated_count and "source_final_label" in evaluated.columns:
        needs_review_mask = evaluated["source_final_label"].astype(str).eq("needs_review")
    else:
        needs_review_mask = pd.Series([False] * evaluated_count)
    confirmed_mask = ~needs_review_mask
    needs_review_eval = evaluated[needs_review_mask].copy() if evaluated_count else evaluated
    confirmed_eval = evaluated[confirmed_mask].copy() if evaluated_count else evaluated
    needs_review_agree = int(needs_review_eval["proxy_agreement"].sum()) if not needs_review_eval.empty else 0
    confirmed_agree = int(confirmed_eval["proxy_agreement"].sum()) if not confirmed_eval.empty else 0
    rows = [
        {"metric": "total_proxy_rows_loaded", "group": "all", "count": proxy_total, "agreement_count": None, "value": proxy_total},
        {"metric": "usable_proxy_rows", "group": "all", "count": usable_total, "agreement_count": None, "value": usable_total},
        {"metric": "non_building_pois_excluded", "group": "proxy_eval_filter", "count": non_building_excluded_total, "agreement_count": None, "value": non_building_excluded_total},
        {"metric": "matched_to_problem2_outputs", "group": "all", "count": matched_total, "agreement_count": None, "value": matched_total},
        {"metric": "matched_building_based_rows", "group": "building_based", "count": building_based_matched_total, "agreement_count": None, "value": building_based_matched_total},
        {"metric": "unmatched_rows", "group": "all", "count": unmatched_total, "agreement_count": None, "value": unmatched_total},
        {"metric": "matched_but_not_evaluated_rows", "group": "all", "count": not_evaluated_matched_total, "agreement_count": None, "value": not_evaluated_matched_total},
        {"metric": "evaluated_rows", "group": "all", "count": evaluated_count, "agreement_count": None, "value": evaluated_count},
        {"metric": "evaluated_building_based_rows", "group": "building_based", "count": evaluated_count, "agreement_count": agreement_count, "value": evaluated_count},
        {
            "metric": "evaluated_confirmed_high_confidence_rows",
            "group": "confirmed_or_high_confidence",
            "count": len(confirmed_eval),
            "agreement_count": confirmed_agree,
            "value": len(confirmed_eval),
        },
        {
            "metric": "evaluated_needs_review_rows",
            "group": "needs_review",
            "count": len(needs_review_eval),
            "agreement_count": needs_review_agree,
            "value": len(needs_review_eval),
        },
        {"metric": "exact_proxy_facade_agreement_count", "group": "all", "count": evaluated_count, "agreement_count": agreement_count, "value": agreement_count},
        {
            "metric": "exact_proxy_facade_agreement_rate",
            "group": "all",
            "count": evaluated_count,
            "agreement_count": agreement_count,
            "value": round(agreement_count / evaluated_count, 6) if evaluated_count else 0.0,
        },
        {
            "metric": "proxy_accuracy_selected_facade_agreement",
            "group": "all",
            "count": evaluated_count,
            "agreement_count": agreement_count,
            "value": round(agreement_count / evaluated_count, 6) if evaluated_count else 0.0,
        },
        {
            "metric": "proxy_accuracy_excluding_needs_review",
            "group": "confirmed_or_high_confidence",
            "count": len(confirmed_eval),
            "agreement_count": confirmed_agree,
            "value": round(confirmed_agree / len(confirmed_eval), 6) if len(confirmed_eval) else 0.0,
        },
        {
            "metric": "proxy_accuracy_needs_review_only",
            "group": "needs_review",
            "count": len(needs_review_eval),
            "agreement_count": needs_review_agree,
            "value": round(needs_review_agree / len(needs_review_eval), 6) if len(needs_review_eval) else 0.0,
        },
        {
            "metric": "baseline_nearest_edge_proxy_accuracy",
            "group": "all",
            "count": evaluated_count,
            "agreement_count": baseline_agreement_count,
            "value": round(baseline_agreement_count / evaluated_count, 6) if evaluated_count else 0.0,
        },
        {
            "metric": "selected_reranked_proxy_accuracy",
            "group": "all",
            "count": evaluated_count,
            "agreement_count": agreement_count,
            "value": round(agreement_count / evaluated_count, 6) if evaluated_count else 0.0,
        },
        {
            "metric": "proxy_coverage_recall_evaluated_over_usable",
            "group": "all",
            "count": max(usable_total - non_building_excluded_total, 0),
            "agreement_count": evaluated_count,
            "value": round(evaluated_count / max(usable_total - non_building_excluded_total, 0), 6) if max(usable_total - non_building_excluded_total, 0) else 0.0,
        },
        {
            "metric": "matched_coverage_matched_over_usable",
            "group": "building_based",
            "count": max(usable_total - non_building_excluded_total, 0),
            "agreement_count": building_based_matched_total,
            "value": round(building_based_matched_total / max(usable_total - non_building_excluded_total, 0), 6) if max(usable_total - non_building_excluded_total, 0) else 0.0,
        },
        {
            "metric": "evaluable_among_matched",
            "group": "building_based",
            "count": building_based_matched_total,
            "agreement_count": evaluated_count,
            "value": round(evaluated_count / building_based_matched_total, 6) if building_based_matched_total else 0.0,
        },
        {
            "metric": "average_selected_facade_distance_to_proxy_point",
            "group": "all",
            "count": evaluated_count,
            "agreement_count": None,
            "value": round(float(evaluated["selected_facade_distance_to_proxy_point_m"].mean()), 3) if evaluated_count else None,
        },
    ]
    rows.extend(_agreement_rate(evaluated, "proxy_point_type"))
    rows.extend(_agreement_rate(evaluated, "selected_method"))
    rows.extend(_agreement_rate(evaluated, "source_dataset"))
    summary = pd.DataFrame(rows)
    summary.to_csv(PROXY_SUMMARY_OUTPUT, index=False)
    return summary


def write_ranker_comparison(evaluated):
    """Compare nearest-edge baseline vs selected/reranked facade agreement."""
    if evaluated.empty:
        comparison = pd.DataFrame(
            [
                {"metric": "evaluated_rows", "count": 0, "rate": 0.0},
                {"metric": "baseline_nearest_edge_agreement", "count": 0, "rate": 0.0},
                {"metric": "selected_reranked_facade_agreement", "count": 0, "rate": 0.0},
                {"metric": "improvement_count", "count": 0, "rate": 0.0},
                {"metric": "worsened_count", "count": 0, "rate": 0.0},
                {"metric": "unchanged_count", "count": 0, "rate": 0.0},
            ]
        )
        comparison.to_csv(PROXY_RANKER_COMPARISON_OUTPUT, index=False)
        return comparison

    baseline = evaluated["nearest_edge_proxy_agreement"].astype(bool)
    selected = evaluated["proxy_agreement"].astype(bool)
    total = len(evaluated)
    improvement = (~baseline & selected)
    worsened = (baseline & ~selected)
    unchanged = baseline == selected
    rows = [
        {"metric": "evaluated_rows", "count": total, "rate": 1.0},
        {"metric": "baseline_nearest_edge_agreement", "count": int(baseline.sum()), "rate": round(float(baseline.mean()), 6)},
        {"metric": "selected_reranked_facade_agreement", "count": int(selected.sum()), "rate": round(float(selected.mean()), 6)},
        {"metric": "improvement_count", "count": int(improvement.sum()), "rate": round(float(improvement.mean()), 6)},
        {"metric": "worsened_count", "count": int(worsened.sum()), "rate": round(float(worsened.mean()), 6)},
        {"metric": "unchanged_count", "count": int(unchanged.sum()), "rate": round(float(unchanged.mean()), 6)},
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(PROXY_RANKER_COMPARISON_OUTPUT, index=False)
    return comparison


def write_failure_analysis_outputs(evaluated):
    """Write easy-to-review rows where the reranker helped or hurt."""
    if evaluated.empty:
        empty = pd.DataFrame()
        for path in [
            PROXY_RERANKER_IMPROVED_ROWS_OUTPUT,
            PROXY_BASELINE_CORRECT_RERANKER_WRONG_ROWS_OUTPUT,
            PROXY_RERANKER_CORRECT_BASELINE_WRONG_OUTPUT,
            PROXY_BASELINE_CORRECT_RERANKER_WRONG_OUTPUT,
            PROXY_NEEDS_REVIEW_EVALUATED_ROWS_OUTPUT,
            PROXY_CONFIRMED_EVALUATED_ROWS_OUTPUT,
            PROXY_EVALUATED_BUILDING_BASED_ROWS_OUTPUT,
        ]:
            empty.to_csv(path, index=False)
        return

    baseline = evaluated["nearest_edge_proxy_agreement"].astype(bool)
    selected = evaluated["proxy_agreement"].astype(bool)
    improved = evaluated[~baseline & selected].copy()
    worsened = evaluated[baseline & ~selected].copy()
    needs_review_mask = evaluated["proxy_eval_included_needs_review"].apply(_as_bool) if "proxy_eval_included_needs_review" in evaluated.columns else evaluated["source_final_label"].astype(str).eq("needs_review")
    needs_review = evaluated[needs_review_mask].copy()
    confirmed = evaluated[~needs_review_mask].copy()
    improved.to_csv(PROXY_RERANKER_IMPROVED_ROWS_OUTPUT, index=False)
    improved.to_csv(PROXY_RERANKER_CORRECT_BASELINE_WRONG_OUTPUT, index=False)
    worsened.to_csv(PROXY_BASELINE_CORRECT_RERANKER_WRONG_ROWS_OUTPUT, index=False)
    worsened.to_csv(PROXY_BASELINE_CORRECT_RERANKER_WRONG_OUTPUT, index=False)
    needs_review.to_csv(PROXY_NEEDS_REVIEW_EVALUATED_ROWS_OUTPUT, index=False)
    confirmed.to_csv(PROXY_CONFIRMED_EVALUATED_ROWS_OUTPUT, index=False)
    evaluated.to_csv(PROXY_EVALUATED_BUILDING_BASED_ROWS_OUTPUT, index=False)
    print(f"Wrote proxy reranker-improved rows: {PROXY_RERANKER_IMPROVED_ROWS_OUTPUT} ({len(improved)} rows)")
    print(f"Wrote proxy reranker-correct/baseline-wrong rows: {PROXY_RERANKER_CORRECT_BASELINE_WRONG_OUTPUT} ({len(improved)} rows)")
    print(
        "Wrote proxy baseline-correct/reranker-wrong rows: "
        f"{PROXY_BASELINE_CORRECT_RERANKER_WRONG_ROWS_OUTPUT} ({len(worsened)} rows)"
    )
    print(f"Wrote proxy needs-review evaluated rows: {PROXY_NEEDS_REVIEW_EVALUATED_ROWS_OUTPUT} ({len(needs_review)} rows)")
    print(f"Wrote proxy confirmed evaluated rows: {PROXY_CONFIRMED_EVALUATED_ROWS_OUTPUT} ({len(confirmed)} rows)")
    print(f"Wrote proxy evaluated building-based rows: {PROXY_EVALUATED_BUILDING_BASED_ROWS_OUTPUT} ({len(evaluated)} rows)")


def _accuracy_pair(df, selected_col):
    """Return count/correct/rate for a boolean agreement column."""
    total = len(df)
    correct = int(df[selected_col].sum()) if total and selected_col in df.columns else 0
    return total, correct, round(correct / total, 6) if total else 0.0


def write_shared_building_logic_outputs(evaluated):
    """Write before/after proxy metrics for optional shared-building logic."""
    if evaluated.empty:
        empty = pd.DataFrame()
        for path in [
            SHARED_BUILDING_LOGIC_EVALUATION_OUTPUT,
            SHARED_BUILDING_LOGIC_SUMMARY_OUTPUT,
            SHARED_BUILDING_IMPROVED_ROWS_OUTPUT,
            SHARED_BUILDING_WORSENED_ROWS_OUTPUT,
            SHARED_BUILDING_UNCHANGED_ROWS_OUTPUT,
        ]:
            empty.to_csv(path, index=False)
        return empty

    out = evaluated.copy()
    out.to_csv(SHARED_BUILDING_LOGIC_EVALUATION_OUTPUT, index=False)

    rows = []
    groups = {
        "overall_building_only": out,
        "shared_building": out[out["is_shared_building"].apply(_as_bool)],
        "non_shared_building": out[~out["is_shared_building"].apply(_as_bool)],
        "possible_mall_or_plaza": out[out["possible_mall_or_plaza"].apply(_as_bool)],
        "non_mall_or_plaza": out[~out["possible_mall_or_plaza"].apply(_as_bool)],
    }
    for group_name, group in groups.items():
        total, baseline_correct, baseline_rate = _accuracy_pair(group, "nearest_edge_proxy_agreement")
        _, before_correct, before_rate = _accuracy_pair(group, "before_shared_logic_proxy_agreement")
        _, after_correct, after_rate = _accuracy_pair(group, "after_shared_logic_proxy_agreement")
        improved = int((group["after_shared_logic_proxy_agreement"].astype(bool) & ~group["before_shared_logic_proxy_agreement"].astype(bool)).sum()) if total else 0
        worsened = int((group["before_shared_logic_proxy_agreement"].astype(bool) & ~group["after_shared_logic_proxy_agreement"].astype(bool)).sum()) if total else 0
        unchanged = int((group["before_shared_logic_proxy_agreement"].astype(bool) == group["after_shared_logic_proxy_agreement"].astype(bool)).sum()) if total else 0
        rows.append(
            {
                "group": group_name,
                "evaluated_rows": total,
                "baseline_correct": baseline_correct,
                "baseline_accuracy": baseline_rate,
                "original_tuned_correct": before_correct,
                "original_tuned_accuracy": before_rate,
                "shared_logic_correct": after_correct,
                "shared_logic_accuracy": after_rate,
                "improved": improved,
                "worsened": worsened,
                "unchanged": unchanged,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(SHARED_BUILDING_LOGIC_SUMMARY_OUTPUT, index=False)

    improved_rows = out[out["shared_logic_improved_proxy_match"].apply(_as_bool)].copy()
    worsened_rows = out[out["shared_logic_worsened_proxy_match"].apply(_as_bool)].copy()
    unchanged_rows = out[~out["shared_logic_improved_proxy_match"].apply(_as_bool) & ~out["shared_logic_worsened_proxy_match"].apply(_as_bool)].copy()
    improved_rows.to_csv(SHARED_BUILDING_IMPROVED_ROWS_OUTPUT, index=False)
    worsened_rows.to_csv(SHARED_BUILDING_WORSENED_ROWS_OUTPUT, index=False)
    unchanged_rows.to_csv(SHARED_BUILDING_UNCHANGED_ROWS_OUTPUT, index=False)
    print(f"Wrote shared-building logic evaluation: {SHARED_BUILDING_LOGIC_EVALUATION_OUTPUT} ({len(out)} rows)")
    print(f"Wrote shared-building logic summary: {SHARED_BUILDING_LOGIC_SUMMARY_OUTPUT}")
    print(f"Wrote shared-building improved/worsened/unchanged rows: {len(improved_rows)}/{len(worsened_rows)}/{len(unchanged_rows)}")
    return summary


def _metric_row(group_name, group):
    """Build an accuracy row for local-storefront summaries."""
    total = len(group)
    baseline_correct = int(group["nearest_edge_proxy_agreement"].sum()) if total else 0
    tuned_correct = int(group["proxy_agreement"].sum()) if total else 0
    local_correct = int(group["local_storefront_proxy_agreement"].sum()) if total else 0
    improved = int((group["local_storefront_proxy_agreement"].astype(bool) & ~group["proxy_agreement"].astype(bool)).sum()) if total else 0
    worsened = int((group["proxy_agreement"].astype(bool) & ~group["local_storefront_proxy_agreement"].astype(bool)).sum()) if total else 0
    unchanged = int((group["proxy_agreement"].astype(bool) == group["local_storefront_proxy_agreement"].astype(bool)).sum()) if total else 0
    return {
        "group": group_name,
        "row_count": total,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": round(baseline_correct / total, 6) if total else 0.0,
        "tuned_correct": tuned_correct,
        "tuned_accuracy_without_local_heuristics": round(tuned_correct / total, 6) if total else 0.0,
        "local_storefront_correct": local_correct,
        "local_storefront_accuracy": round(local_correct / total, 6) if total else 0.0,
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
    }


def _point_from_row(row, lat_col, lon_col, metric_crs):
    """Project a row point to the metric CRS."""
    lat = pd.to_numeric(pd.Series([row.get(lat_col)]), errors="coerce").iloc[0]
    lon = pd.to_numeric(pd.Series([row.get(lon_col)]), errors="coerce").iloc[0]
    if pd.isna(lat) or pd.isna(lon):
        return None
    return gpd.GeoDataFrame(
        [{"geometry": Point(float(lon), float(lat))}],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_crs(metric_crs).iloc[0].geometry


def _local_consistency_score(poi_distance, proxy_distance):
    """Score candidates where POI and proxy point support the same facade."""
    gap = abs(float(poi_distance) - float(proxy_distance))
    return distance_score(poi_distance) * distance_score(proxy_distance) * (1.0 / (1.0 + gap))


def write_local_storefront_outputs(evaluated, building_lookup, metric_crs):
    """Run optional local storefront heuristic analysis for shared buildings."""
    if evaluated.empty or not ENABLE_LOCAL_STOREFRONT_HEURISTICS:
        empty = pd.DataFrame()
        for path in [
            LOCAL_STOREFRONT_EVALUATION_OUTPUT,
            LOCAL_STOREFRONT_SUMMARY_OUTPUT,
            LOCAL_STOREFRONT_IMPROVED_ROWS_OUTPUT,
            LOCAL_STOREFRONT_WORSENED_ROWS_OUTPUT,
            CORNER_AMBIGUITY_ANALYSIS_OUTPUT,
        ]:
            empty.to_csv(path, index=False)
        return empty

    density_counts = {}
    building_totals = evaluated["building_id"].astype(str).value_counts().to_dict()
    for _, row in evaluated.iterrows():
        building_id = _clean_id(row.get("building_id"))
        proxy_edge_id = _clean_id(row.get("proxy_facade_edge_id"))
        if building_id and proxy_edge_id:
            density_counts.setdefault(building_id, {})
            density_counts[building_id][proxy_edge_id] = density_counts[building_id].get(proxy_edge_id, 0) + 1

    local_rows = []
    for _, row in evaluated.iterrows():
        building_id = _clean_id(row.get("building_id"))
        if not building_id or building_id not in building_lookup.index:
            continue
        poi_point = _point_from_row(row, "poi_lat", "poi_lon", metric_crs)
        proxy_point = _point_from_row(row, "proxy_lat", "proxy_lon", metric_crs)
        if poi_point is None or proxy_point is None:
            continue

        facades = _facade_lookup(building_id, building_lookup.loc[building_id].geometry)
        total_proxy_points = int(building_totals.get(building_id, 0))
        facade_density = density_counts.get(building_id, {})
        dominant_facade = max(facade_density, key=facade_density.get) if facade_density else ""
        selected_edge = _clean_id(row.get("selected_facade_edge_id"))
        nearest_edge = _clean_id(row.get("nearest_edge_id"))
        nearest_distance = None
        best = None
        best_score = None

        for edge_id, facade in facades.items():
            poi_distance = float(poi_point.distance(facade["geometry"]))
            proxy_distance = float(proxy_point.distance(facade["geometry"]))
            if nearest_distance is None or poi_distance < nearest_distance:
                nearest_distance = poi_distance
            density_count = int(facade_density.get(edge_id, 0))
            density_score = density_count / total_proxy_points if total_proxy_points else 0.0
            consistency = _local_consistency_score(poi_distance, proxy_distance)
            score = (
                0.75 * distance_score(poi_distance)
                + BUILDING_SIDE_DENSITY_WEIGHT * density_score
                + POI_PROXY_CONSISTENCY_WEIGHT * consistency
                - 0.20 * (0.25 if poi_distance <= FACADE_DENSITY_BUFFER_M and row.get("is_corner_ambiguous") else 0.0)
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "edge_id": edge_id,
                    "poi_distance": poi_distance,
                    "proxy_distance": proxy_distance,
                    "density_count": density_count,
                    "density_score": density_score,
                    "consistency": consistency,
                    "score": score,
                }

        # Guardrail: keep the original tuned selection if local evidence is not clearly better.
        original_facade = facades.get(selected_edge)
        if original_facade is not None and best is not None:
            original_poi_distance = float(poi_point.distance(original_facade["geometry"]))
            original_proxy_distance = float(proxy_point.distance(original_facade["geometry"]))
            original_density = int(facade_density.get(selected_edge, 0))
            original_density_score = original_density / total_proxy_points if total_proxy_points else 0.0
            original_consistency = _local_consistency_score(original_poi_distance, original_proxy_distance)
            original_score = (
                0.75 * distance_score(original_poi_distance)
                + BUILDING_SIDE_DENSITY_WEIGHT * original_density_score
                + POI_PROXY_CONSISTENCY_WEIGHT * original_consistency
            )
            if best["edge_id"] != selected_edge and best["score"] < original_score + 0.10:
                best = {
                    "edge_id": selected_edge,
                    "poi_distance": original_poi_distance,
                    "proxy_distance": original_proxy_distance,
                    "density_count": original_density,
                    "density_score": original_density_score,
                    "consistency": original_consistency,
                    "score": original_score,
                }

        if best is None:
            continue
        proxy_edge = _clean_id(row.get("proxy_facade_edge_id"))
        local_correct = best["edge_id"] == proxy_edge
        tuned_correct = _as_bool(row.get("proxy_agreement"))
        local = row.to_dict()
        local.update(
            {
                "local_storefront_facade_edge_id": best["edge_id"],
                "local_storefront_proxy_agreement": bool(local_correct),
                "building_side_density_count": best["density_count"],
                "building_side_density_score": round(best["density_score"], 6),
                "building_total_proxy_points": total_proxy_points,
                "dominant_building_facade_id": dominant_facade,
                "is_on_dominant_building_side": bool(best["edge_id"] == dominant_facade),
                "poi_to_candidate_facade_distance_m": round(best["poi_distance"], 3),
                "proxy_to_candidate_facade_distance_m": round(best["proxy_distance"], 3),
                "poi_proxy_facade_distance_gap_m": round(abs(best["poi_distance"] - best["proxy_distance"]), 3),
                "poi_proxy_consistency_score": round(best["consistency"], 6),
                "local_storefront_score": round(best["score"], 6),
                "local_storefront_improved_proxy_match": bool(local_correct and not tuned_correct),
                "local_storefront_worsened_proxy_match": bool(tuned_correct and not local_correct),
            }
        )
        local_rows.append(local)

    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(LOCAL_STOREFRONT_EVALUATION_OUTPUT, index=False)
    local_df.to_csv(CORNER_AMBIGUITY_ANALYSIS_OUTPUT, index=False)
    improved = local_df[local_df["local_storefront_improved_proxy_match"].astype(bool)].copy() if not local_df.empty else local_df
    worsened = local_df[local_df["local_storefront_worsened_proxy_match"].astype(bool)].copy() if not local_df.empty else local_df
    improved.to_csv(LOCAL_STOREFRONT_IMPROVED_ROWS_OUTPUT, index=False)
    worsened.to_csv(LOCAL_STOREFRONT_WORSENED_ROWS_OUTPUT, index=False)

    rows = []
    groups = {
        "overall_building_only": local_df,
        "shared_building": local_df[local_df["is_shared_building"].astype(bool)] if not local_df.empty else local_df,
        "possible_mall_or_plaza": local_df[local_df["possible_mall_or_plaza"].astype(bool)] if not local_df.empty else local_df,
        "non_shared_building": local_df[~local_df["is_shared_building"].astype(bool)] if not local_df.empty else local_df,
    }
    for name, group in groups.items():
        rows.append(_metric_row(name, group))

    dominant = local_df[local_df["is_on_dominant_building_side"].astype(bool)] if not local_df.empty else local_df
    not_dominant = local_df[~local_df["is_on_dominant_building_side"].astype(bool)] if not local_df.empty else local_df
    rows.append({"group": "predicted_on_dominant_building_side", **_metric_row("predicted_on_dominant_building_side", dominant)})
    rows.append({"group": "predicted_not_on_dominant_building_side", **_metric_row("predicted_not_on_dominant_building_side", not_dominant)})

    correct = local_df[local_df["local_storefront_proxy_agreement"].astype(bool)] if not local_df.empty else local_df
    wrong = local_df[~local_df["local_storefront_proxy_agreement"].astype(bool)] if not local_df.empty else local_df
    corner_ambiguous = local_df[local_df["is_corner_ambiguous"].astype(bool)] if not local_df.empty else local_df
    near_correct_count = int(local_df["near_correct_under_corner_rule"].sum()) if not local_df.empty else 0
    strict_correct = int(local_df["proxy_agreement"].sum()) if not local_df.empty else 0
    rows.extend(
        [
            {
                "group": "poi_proxy_consistency_correct_predictions",
                "row_count": len(correct),
                "average_poi_proxy_consistency_score": round(float(correct["poi_proxy_consistency_score"].mean()), 6) if len(correct) else None,
            },
            {
                "group": "poi_proxy_consistency_wrong_predictions",
                "row_count": len(wrong),
                "average_poi_proxy_consistency_score": round(float(wrong["poi_proxy_consistency_score"].mean()), 6) if len(wrong) else None,
            },
            {
                "group": "corner_ambiguity",
                "row_count": len(corner_ambiguous),
                "strict_proxy_accuracy": round(strict_correct / len(local_df), 6) if len(local_df) else 0.0,
                "corner_tolerant_near_correct_accuracy": round(near_correct_count / len(local_df), 6) if len(local_df) else 0.0,
                "additional_near_correct_only_under_corner_rule": max(0, near_correct_count - strict_correct),
            },
        ]
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(LOCAL_STOREFRONT_SUMMARY_OUTPUT, index=False)
    print(f"Wrote local storefront heuristic evaluation: {LOCAL_STOREFRONT_EVALUATION_OUTPUT} ({len(local_df)} rows)")
    print(f"Wrote local storefront heuristic summary: {LOCAL_STOREFRONT_SUMMARY_OUTPUT}")
    return summary


def run_proxy_evaluation():
    """Run full proxy matching and facade evaluation."""
    p2, proxy, p1 = _load_inputs()
    matched, unmatched = match_proxy_to_problem2(p2, proxy, p1)
    eligible_matched, excluded_non_building = split_building_based_matches(matched)
    buildings, _ = load_buildings()
    buildings_m = buildings.to_crs(buildings.estimate_utm_crs() or "EPSG:3857")
    building_lookup = buildings_m.drop_duplicates("building_id").set_index("building_id")

    evaluated_rows = []
    skipped_rows = []
    for _, row in eligible_matched.iterrows():
        evaluated, skip_reason = _evaluate_one(row, building_lookup, buildings_m.crs)
        if evaluated is None:
            skipped = row.to_dict()
            skipped["unmatched_reason"] = skip_reason
            skipped_rows.append(skipped)
            continue
        evaluated_rows.append(evaluated)

    pd.DataFrame(skipped_rows).to_csv(PROXY_MATCHED_NOT_EVALUATED_OUTPUT, index=False)
    print(f"Wrote proxy matched-but-not-evaluated rows: {PROXY_MATCHED_NOT_EVALUATED_OUTPUT} ({len(skipped_rows)} rows)")

    evaluated = pd.DataFrame(evaluated_rows)
    PROXY_EVALUATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(PROXY_EVALUATION_OUTPUT, index=False)
    raw_proxy_total = count_raw_proxy_rows()
    summary = write_summary(
        evaluated,
        proxy_total=raw_proxy_total,
        usable_total=len(proxy),
        matched_total=len(matched),
        unmatched_total=len(unmatched),
        not_evaluated_matched_total=len(skipped_rows),
        non_building_excluded_total=len(excluded_non_building),
        building_based_matched_total=len(eligible_matched),
    )
    comparison = write_ranker_comparison(evaluated)
    write_failure_analysis_outputs(evaluated)
    shared_summary = write_shared_building_logic_outputs(evaluated)
    local_summary = write_local_storefront_outputs(evaluated, building_lookup, buildings_m.crs)

    print(f"Wrote proxy matched rows: {PROXY_MATCHED_ROWS_OUTPUT} ({len(matched)} rows)")
    print(f"Wrote proxy unmatched rows: {PROXY_UNMATCHED_ROWS_OUTPUT} ({len(unmatched)} rows)")
    print(f"Wrote proxy evaluation: {PROXY_EVALUATION_OUTPUT} ({len(evaluated)} rows)")
    print(f"Wrote proxy summary: {PROXY_SUMMARY_OUTPUT}")
    print(f"Wrote proxy ranker comparison: {PROXY_RANKER_COMPARISON_OUTPUT}")
    if shared_summary is not None and not shared_summary.empty:
        print("\nShared-building logic summary:")
        print(shared_summary.to_string(index=False))
    if local_summary is not None and not local_summary.empty:
        print("\nLocal storefront heuristic summary:")
        print(local_summary.to_string(index=False))
    if not summary.empty:
        rate = summary[summary["metric"] == "exact_proxy_facade_agreement_rate"]["value"]
        print(f"Proxy agreement rate: {rate.iloc[0] if not rate.empty else 0.0}")
    return evaluated, summary, comparison


if __name__ == "__main__":
    run_proxy_evaluation()
