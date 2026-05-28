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
    EXCLUDE_NON_BUILDING_POIS_FROM_PROXY_EVAL,
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
)
from proxy_benchmark_builder import count_raw_proxy_rows
from run_pipeline import extract_facades, load_buildings


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
    proxy_edge_id = min(distances, key=distances.get)
    selected_edge_id = _clean_id(row.get("selected_facade_edge_id"))
    nearest_edge_id = _clean_id(row.get("nearest_edge_id"))
    selected_facade = facades.get(selected_edge_id)
    nearest_facade = facades.get(nearest_edge_id)

    selected_distance = proxy_point.distance(selected_facade["geometry"]) if selected_facade else None
    nearest_distance = proxy_point.distance(nearest_facade["geometry"]) if nearest_facade else None
    proxy_distance = distances[proxy_edge_id]
    selected_agreement = bool(selected_edge_id and selected_edge_id == proxy_edge_id)
    nearest_agreement = bool(nearest_edge_id and nearest_edge_id == proxy_edge_id)

    result = {
        "poi_id": row.get("poi_id"),
        "poi_name": row.get("poi_name"),
        "source_dataset": row.get("proxy_source_dataset"),
        "proxy_point_type": row.get("proxy_proxy_point_type"),
        "source_final_label": row.get("source_final_label", row.get("final_label", row.get("problem1_final_label"))),
        "problem2_eval_mode": row.get("problem2_eval_mode", "strict"),
        "proxy_eval_included_needs_review": _as_bool(row.get("proxy_eval_included_needs_review")),
        "selected_facade_edge_id": selected_edge_id,
        "nearest_edge_id": nearest_edge_id,
        "proxy_facade_edge_id": proxy_edge_id,
        "proxy_agreement": selected_agreement,
        "nearest_edge_proxy_agreement": nearest_agreement,
        "selected_facade_distance_to_proxy_point_m": round(float(selected_distance), 3) if selected_distance is not None else None,
        "nearest_edge_distance_to_proxy_point_m": round(float(nearest_distance), 3) if nearest_distance is not None else None,
        "proxy_point_distance_to_proxy_edge_m": round(float(proxy_distance), 3),
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

    print(f"Wrote proxy matched rows: {PROXY_MATCHED_ROWS_OUTPUT} ({len(matched)} rows)")
    print(f"Wrote proxy unmatched rows: {PROXY_UNMATCHED_ROWS_OUTPUT} ({len(unmatched)} rows)")
    print(f"Wrote proxy evaluation: {PROXY_EVALUATION_OUTPUT} ({len(evaluated)} rows)")
    print(f"Wrote proxy summary: {PROXY_SUMMARY_OUTPUT}")
    print(f"Wrote proxy ranker comparison: {PROXY_RANKER_COMPARISON_OUTPUT}")
    if not summary.empty:
        rate = summary[summary["metric"] == "exact_proxy_facade_agreement_rate"]["value"]
        print(f"Proxy agreement rate: {rate.iloc[0] if not rate.empty else 0.0}")
    return evaluated, summary, comparison


if __name__ == "__main__":
    run_proxy_evaluation()
