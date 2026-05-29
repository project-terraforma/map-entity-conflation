"""Analyze and tune the Problem 2 facade reranker with proxy storefront points.

The benchmark uses entrance/sign/front-door points as proxy facade labels. These
are useful agreement metrics, not human-verified ground-truth facade labels.
"""

import itertools
import json

import pandas as pd
from shapely.geometry import Point

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover
    raise ImportError("Reranker tuning requires geopandas.") from exc

from config import (
    BEST_RERANKER_CONFIG_OUTPUT,
    BEST_RERANKER_CONFIG_TRAIN_TEST_OUTPUT,
    BEST_LOCAL_STOREFRONT_CONFIG_OUTPUT,
    LOCAL_STOREFRONT_TUNING_RESULTS_OUTPUT,
    PROXY_ACCURACY_BY_LABEL_OUTPUT,
    PROXY_CLEAN_SUBSET_EXCLUSIONS_OUTPUT,
    PROXY_CLEAN_SUBSET_METRICS_OUTPUT,
    PROXY_CLEAN_SUBSET_ROWS_OUTPUT,
    PROXY_SHARED_BUILDING_ANALYSIS_OUTPUT,
    RERANKER_TUNING_RESULTS_OUTPUT,
    RERANKER_TUNING_TRAIN_TEST_RESULTS_OUTPUT,
)
from proxy_facade_evaluator import (
    _as_bool,
    _clean_id,
    _load_inputs,
    match_proxy_to_problem2,
    split_building_based_matches,
)
from run_pipeline import (
    distance_score,
    edge_candidate,
    extract_facades,
    load_buildings,
    load_streets,
)

HIGH_CONFIDENCE_LABELS = {
    "same_building_confirmed",
    "building_validated_candidate",
    "spatial_address_candidate",
    "text_and_distance_match",
}

MALL_OR_PLAZA_KEYWORDS = {
    "mall",
    "plaza",
    "shopping center",
    "shopping centre",
    "outlet",
    "market",
    "center",
    "centre",
    "square",
    "suite",
    "unit",
    "store",
}


def _proxy_point(row, metric_crs):
    """Project a proxy point to the metric CRS."""
    proxy_lat = pd.to_numeric(pd.Series([row.get("proxy_proxy_lat")]), errors="coerce").iloc[0]
    proxy_lon = pd.to_numeric(pd.Series([row.get("proxy_proxy_lon")]), errors="coerce").iloc[0]
    if pd.isna(proxy_lat) or pd.isna(proxy_lon):
        return None
    return gpd.GeoDataFrame(
        [{"geometry": Point(float(proxy_lon), float(proxy_lat))}],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_crs(metric_crs).iloc[0].geometry


def _poi_point(row, metric_crs):
    """Project the POI point to the metric CRS."""
    poi_lat = pd.to_numeric(pd.Series([row.get("poi_lat")]), errors="coerce").iloc[0]
    poi_lon = pd.to_numeric(pd.Series([row.get("poi_lon")]), errors="coerce").iloc[0]
    if pd.isna(poi_lat) or pd.isna(poi_lon):
        return None
    return gpd.GeoDataFrame(
        [{"geometry": Point(float(poi_lon), float(poi_lat))}],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_crs(metric_crs).iloc[0].geometry


def _contains_keyword(row, keywords):
    """Return True if POI text contains any shared-building keyword."""
    text = " ".join(
        str(row.get(col, "") or "").lower()
        for col in ["poi_name", "poi_types", "proxy_notes", "matched_address_text"]
    )
    return any(keyword in text for keyword in keywords)


def _build_records():
    """Create candidate records for eligible building-based proxy rows."""
    p2, proxy, p1 = _load_inputs()
    matched, unmatched = match_proxy_to_problem2(p2, proxy, p1)
    eligible, excluded = split_building_based_matches(matched)

    building_counts = eligible["building_id"].astype(str).value_counts().to_dict() if not eligible.empty else {}

    buildings, _ = load_buildings()
    buildings_m = buildings.to_crs(buildings.estimate_utm_crs() or "EPSG:3857")
    streets_m, _ = load_streets(buildings_m.crs)
    street_available = streets_m is not None and not streets_m.empty
    building_lookup = buildings_m.drop_duplicates("building_id").set_index("building_id")

    rows = []
    nearest_edge_ids = []
    skipped = []
    for _, row in eligible.iterrows():
        if str(row.get("problem2_status", "")) != "facade_candidate":
            skipped.append({**row.to_dict(), "skip_reason": "problem2_row_has_no_selected_facade"})
            continue
        building_id = _clean_id(row.get("building_id"))
        if not building_id or building_id not in building_lookup.index:
            skipped.append({**row.to_dict(), "skip_reason": "missing_building_geometry"})
            continue
        poi = _poi_point(row, buildings_m.crs)
        proxy_point = _proxy_point(row, buildings_m.crs)
        if poi is None or proxy_point is None:
            skipped.append({**row.to_dict(), "skip_reason": "missing_coordinates"})
            continue

        facades = extract_facades(building_id, building_lookup.loc[building_id].geometry)
        candidates = [edge_candidate(row, poi, facade, streets_m) for facade in facades]
        if not candidates:
            skipped.append({**row.to_dict(), "skip_reason": "no_facade_edges"})
            continue

        proxy_distances = {
            candidate["edge_id"]: float(proxy_point.distance(candidate["edge_geometry"]))
            for candidate in candidates
        }
        proxy_edge_id = min(proxy_distances, key=proxy_distances.get)
        nearest = min(candidates, key=lambda item: item["distance_m"])
        nearest_edge_ids.append(nearest["edge_id"])
        shared_building_poi_count = int(building_counts.get(building_id, 0))
        possible_mall_or_plaza = shared_building_poi_count >= 5 or _contains_keyword(row, MALL_OR_PLAZA_KEYWORDS)

        rows.append(
            {
                "poi_id": row.get("poi_id"),
                "poi_name": row.get("poi_name"),
                "source_dataset": row.get("proxy_source_dataset"),
                "source_final_label": row.get("source_final_label", row.get("final_label")),
                "proxy_eval_included_needs_review": _as_bool(row.get("proxy_eval_included_needs_review")),
                "building_id": building_id,
                "shared_building_poi_count": shared_building_poi_count,
                "is_shared_building": shared_building_poi_count > 1,
                "possible_mall_or_plaza": possible_mall_or_plaza,
                "nearest_edge_id": nearest["edge_id"],
                "proxy_facade_edge_id": proxy_edge_id,
                "proxy_point_to_baseline_facade_distance": round(proxy_distances[nearest["edge_id"]], 3),
                "proxy_point_distance_to_proxy_edge": round(proxy_distances[proxy_edge_id], 3),
                "proxy_distances": proxy_distances,
                "candidates": candidates,
            }
        )

    shared_counts = pd.Series(nearest_edge_ids).value_counts().to_dict()
    context = {
        "usable_proxy_total": len(proxy),
        "matched_total": len(matched),
        "unmatched_total": len(unmatched),
        "non_building_excluded_total": len(excluded),
        "skipped_total": len(skipped),
        "street_available": street_available,
    }
    return rows, context, shared_counts


def _score(candidate, shared_count, params):
    """Compute a distance-dominant candidate score for tuning."""
    shared_penalty = 0.10 if shared_count >= 5 else 0.0
    return (
        params["distance_weight"] * distance_score(candidate["distance_m"])
        + params.get("street_alignment_weight", 0.0) * candidate["street_alignment_score"]
        + params.get("street_facing_weight", 0.0) * candidate["street_facing_score"]
        + params.get("edge_length_weight", 0.0) * candidate["edge_length_score"]
        - params["corner_penalty_weight"] * candidate["corner_penalty"]
        - params["shared_facade_penalty_weight"] * shared_penalty
    )


def _select(candidates, nearest_edge_id, shared_counts, params):
    """Select a candidate with distance-ratio and close-nearest guardrails."""
    nearest = next(candidate for candidate in candidates if candidate["edge_id"] == nearest_edge_id)
    nearest_distance = max(float(nearest["distance_m"]), 0.001)
    scored = [
        {
            **candidate,
            "score": _score(candidate, shared_counts.get(candidate["edge_id"], 0), params),
        }
        for candidate in candidates
    ]
    eligible = [
        candidate
        for candidate in scored
        if float(candidate["distance_m"]) <= nearest_distance * params["max_distance_ratio_for_override"]
    ]
    selected = max(eligible or scored, key=lambda item: item["score"])
    nearest_scored = next(candidate for candidate in scored if candidate["edge_id"] == nearest_edge_id)
    if nearest_distance <= params["lockin_distance_m"]:
        margin = params["strong_override_margin"]
        if selected["edge_id"] != nearest_edge_id and selected["score"] < nearest_scored["score"] + margin:
            return nearest_edge_id
    return selected["edge_id"]


def _selected_rows(records, params, shared_counts):
    """Return row-level selections for one parameter combination."""
    rows = []
    for record in records:
        selected_edge_id = _select(record["candidates"], record["nearest_edge_id"], shared_counts, params)
        proxy_edge_id = record["proxy_facade_edge_id"]
        baseline_agree = record["nearest_edge_id"] == proxy_edge_id
        selected_agree = selected_edge_id == proxy_edge_id
        rows.append(
            {
                "poi_id": record["poi_id"],
                "poi_name": record["poi_name"],
                "source_dataset": record["source_dataset"],
                "source_final_label": record["source_final_label"],
                "label_group": _label_group(record["source_final_label"]),
                "building_id": record["building_id"],
                "shared_building_poi_count": record["shared_building_poi_count"],
                "is_shared_building": record["is_shared_building"],
                "possible_mall_or_plaza": record["possible_mall_or_plaza"],
                "nearest_edge_id": record["nearest_edge_id"],
                "selected_facade_edge_id": selected_edge_id,
                "proxy_facade_edge_id": proxy_edge_id,
                "baseline_correct": baseline_agree,
                "selected_correct": selected_agree,
                "proxy_point_to_selected_facade_distance": round(record["proxy_distances"].get(selected_edge_id, float("nan")), 3),
                "proxy_point_to_baseline_facade_distance": record["proxy_point_to_baseline_facade_distance"],
            }
        )
    return pd.DataFrame(rows)


def _metrics_for_rows(rows):
    """Return selected-vs-baseline metrics for a row DataFrame."""
    total = len(rows)
    selected_correct = int(rows["selected_correct"].sum()) if total else 0
    baseline_correct = int(rows["baseline_correct"].sum()) if total else 0
    return {
        "row_count": total,
        "selected_correct": selected_correct,
        "selected_proxy_accuracy": round(selected_correct / total, 6) if total else 0.0,
        "baseline_correct": baseline_correct,
        "baseline_proxy_accuracy": round(baseline_correct / total, 6) if total else 0.0,
        "difference_vs_baseline": round((selected_correct - baseline_correct) / total, 6) if total else 0.0,
    }


def _evaluate_params(records, params, context, shared_counts):
    """Evaluate one parameter combination."""
    rows = _selected_rows(records, params, shared_counts)
    metrics = _metrics_for_rows(rows)
    improved = int((rows["selected_correct"] & ~rows["baseline_correct"]).sum()) if not rows.empty else 0
    worsened = int((rows["baseline_correct"] & ~rows["selected_correct"]).sum()) if not rows.empty else 0
    unchanged = int((rows["baseline_correct"] == rows["selected_correct"]).sum()) if not rows.empty else 0
    eligible_usable = max(context["usable_proxy_total"] - context["non_building_excluded_total"], 0)
    needs_review = rows[rows["source_final_label"].astype(str).eq("needs_review")]
    confirmed = rows[rows["source_final_label"].isin(HIGH_CONFIDENCE_LABELS)]
    return {
        **params,
        "evaluated_building_based_rows": metrics["row_count"],
        "non_building_pois_excluded": context["non_building_excluded_total"],
        "selected_reranked_proxy_accuracy": metrics["selected_proxy_accuracy"],
        "baseline_nearest_edge_proxy_accuracy": metrics["baseline_proxy_accuracy"],
        "improvement_over_baseline": metrics["difference_vs_baseline"],
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "proxy_coverage_recall": round(metrics["row_count"] / eligible_usable, 6) if eligible_usable else 0.0,
        "confirmed_high_confidence_accuracy": _metrics_for_rows(confirmed)["selected_proxy_accuracy"],
        "needs_review_accuracy": _metrics_for_rows(needs_review)["selected_proxy_accuracy"],
        "boulder_evaluated_rows": int((rows["source_dataset"] == "boulder").sum()) if not rows.empty else 0,
        "louisville_evaluated_rows": int((rows["source_dataset"] == "louisville").sum()) if not rows.empty else 0,
    }


def _label_group(label):
    """Map final labels to summary groups."""
    if label in HIGH_CONFIDENCE_LABELS:
        return "high_confidence_building_linked"
    if label == "needs_review":
        return "needs_review_only"
    return "other_building_based"


def _parameter_grid():
    """Return the expanded guardrail/proxy-distance tuning grid."""
    return {
        "distance_weight": [0.75, 0.8, 0.85, 0.9],
        "street_alignment_weight": [0.0, 0.03, 0.06, 0.09],
        "street_facing_weight": [0.0],
        "edge_length_weight": [0.0],
        "corner_penalty_weight": [0.0, 0.1, 0.2],
        "shared_facade_penalty_weight": [0.0, 0.1, 0.2],
        "lockin_distance_m": [3.0, 5.0, 7.5, 10.0],
        "max_distance_ratio_for_override": [1.25, 1.5, 2.0],
        "strong_override_margin": [0.1, 0.15, 0.2],
    }


def _iter_params():
    """Yield parameter combinations."""
    grid = _parameter_grid()
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield {key: value for key, value in zip(keys, values)}


def _best_from_results(results_df):
    """Choose the best transparent tuning row."""
    if results_df.empty:
        return {}
    ordered = results_df.sort_values(
        ["selected_reranked_proxy_accuracy", "improvement_over_baseline", "worsened"],
        ascending=[False, False, True],
    )
    return ordered.iloc[0].to_dict()


def _config_from_result(result):
    """Extract reusable config keys from a result row."""
    keys = [
        "distance_weight",
        "street_alignment_weight",
        "street_facing_weight",
        "edge_length_weight",
        "corner_penalty_weight",
        "shared_facade_penalty_weight",
        "lockin_distance_m",
        "max_distance_ratio_for_override",
        "strong_override_margin",
    ]
    config = {key: result[key] for key in keys if key in result}
    config["proxy_accuracy"] = result.get("selected_reranked_proxy_accuracy")
    config["baseline_proxy_accuracy"] = result.get("baseline_nearest_edge_proxy_accuracy")
    config["improvement_over_baseline"] = result.get("improvement_over_baseline")
    return config


def _write_accuracy_by_label(selected_rows):
    """Write accuracy by individual labels and grouped categories."""
    rows = []
    for label, group in selected_rows.groupby("source_final_label", dropna=False):
        rows.append({"group_type": "source_final_label", "group": str(label), **_metrics_for_rows(group)})

    group_defs = {
        "high_confidence_building_linked": selected_rows[selected_rows["source_final_label"].isin(HIGH_CONFIDENCE_LABELS)],
        "needs_review_only": selected_rows[selected_rows["source_final_label"].astype(str).eq("needs_review")],
        "all_building_based": selected_rows,
    }
    for group_name, group in group_defs.items():
        rows.append({"group_type": "combined_category", "group": group_name, **_metrics_for_rows(group)})

    output = pd.DataFrame(rows)
    output.to_csv(PROXY_ACCURACY_BY_LABEL_OUTPUT, index=False)
    return output


def _write_shared_analysis(selected_rows):
    """Write shared-building row analysis and metrics."""
    rows = selected_rows.copy()
    rows.to_csv(PROXY_SHARED_BUILDING_ANALYSIS_OUTPUT, index=False)

    summary_rows = []
    groups = {
        "shared_building_rows": rows[rows["is_shared_building"].astype(bool)],
        "non_shared_building_rows": rows[~rows["is_shared_building"].astype(bool)],
        "possible_mall_or_plaza_rows": rows[rows["possible_mall_or_plaza"].astype(bool)],
        "non_mall_or_plaza_rows": rows[~rows["possible_mall_or_plaza"].astype(bool)],
    }
    for name, group in groups.items():
        summary_rows.append({"group_type": "shared_building_analysis", "group": name, **_metrics_for_rows(group)})
    return pd.DataFrame(summary_rows)


def _write_clean_subset_outputs(selected_rows, context):
    """Write clean-subset rows, exclusions, and metric summaries."""
    exclusions = []
    clean_mask = pd.Series(True, index=selected_rows.index)

    mall_mask = selected_rows["possible_mall_or_plaza"].astype(bool)
    for _, row in selected_rows[mall_mask].iterrows():
        exclusions.append({**row.to_dict(), "exclusion_reason": "possible_mall_or_plaza_or_shared_commercial_building"})
    clean_mask &= ~mall_mask

    clean_rows = selected_rows[clean_mask].copy()
    clean_high_conf = clean_rows[clean_rows["source_final_label"].isin(HIGH_CONFIDENCE_LABELS)].copy()
    high_conf = selected_rows[selected_rows["source_final_label"].isin(HIGH_CONFIDENCE_LABELS)].copy()

    clean_rows.to_csv(PROXY_CLEAN_SUBSET_ROWS_OUTPUT, index=False)
    pd.DataFrame(exclusions).to_csv(PROXY_CLEAN_SUBSET_EXCLUSIONS_OUTPUT, index=False)

    metric_rows = [
        {"subset": "all_building_based", "exclusion_note": "none", **_metrics_for_rows(selected_rows)},
        {"subset": "high_confidence_building_linked", "exclusion_note": "none", **_metrics_for_rows(high_conf)},
        {"subset": "clean_building_proxy_subset", "exclusion_note": "excluded possible mall/plaza/shared-commercial rows", **_metrics_for_rows(clean_rows)},
        {"subset": "clean_high_confidence_subset", "exclusion_note": "high-confidence labels and excluded possible mall/plaza/shared-commercial rows", **_metrics_for_rows(clean_high_conf)},
        {
            "subset": "excluded_non_building_pois",
            "exclusion_note": "excluded before facade accuracy because POIs are not building-based",
            "row_count": context["non_building_excluded_total"],
            "selected_correct": None,
            "selected_proxy_accuracy": None,
            "baseline_correct": None,
            "baseline_proxy_accuracy": None,
            "difference_vs_baseline": None,
        },
        {
            "subset": "excluded_possible_mall_or_plaza",
            "exclusion_note": "reported separately, not removed silently",
            "row_count": int(mall_mask.sum()),
            "selected_correct": None,
            "selected_proxy_accuracy": None,
            "baseline_correct": None,
            "baseline_proxy_accuracy": None,
            "difference_vs_baseline": None,
        },
    ]
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(PROXY_CLEAN_SUBSET_METRICS_OUTPUT, index=False)
    return metrics


def _train_test_split(records):
    """Return deterministic 70/30 train/test records."""
    frame = pd.DataFrame({"idx": range(len(records))}).sample(frac=1.0, random_state=42).reset_index(drop=True)
    split = int(len(frame) * 0.70)
    train_idx = set(frame.iloc[:split]["idx"].tolist())
    train = [record for idx, record in enumerate(records) if idx in train_idx]
    test = [record for idx, record in enumerate(records) if idx not in train_idx]
    return train, test


def _train_test_tuning(records, context, shared_counts):
    """Tune on train rows and report the selected config on held-out test rows."""
    train_records, test_records = _train_test_split(records)
    rows = []
    for params in _iter_params():
        train_result = _evaluate_params(train_records, params, context, shared_counts)
        test_result = _evaluate_params(test_records, params, context, shared_counts)
        rows.append(
            {
                **params,
                "train_rows": train_result["evaluated_building_based_rows"],
                "test_rows": test_result["evaluated_building_based_rows"],
                "train_accuracy": train_result["selected_reranked_proxy_accuracy"],
                "test_accuracy": test_result["selected_reranked_proxy_accuracy"],
                "train_baseline_accuracy": train_result["baseline_nearest_edge_proxy_accuracy"],
                "test_baseline_accuracy": test_result["baseline_nearest_edge_proxy_accuracy"],
                "test_improvement_over_baseline": test_result["improvement_over_baseline"],
                "test_improved": test_result["improved"],
                "test_worsened": test_result["worsened"],
                "test_unchanged": test_result["unchanged"],
                "high_confidence_test_accuracy": test_result["confirmed_high_confidence_accuracy"],
                "needs_review_test_accuracy": test_result["needs_review_accuracy"],
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["train_accuracy", "test_improvement_over_baseline", "test_worsened"],
        ascending=[False, False, True],
    )
    results.to_csv(RERANKER_TUNING_TRAIN_TEST_RESULTS_OUTPUT, index=False)
    best = results.iloc[0].to_dict() if not results.empty else {}
    with open(BEST_RERANKER_CONFIG_TRAIN_TEST_OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(_config_from_result({
            **best,
            "selected_reranked_proxy_accuracy": best.get("test_accuracy"),
            "baseline_nearest_edge_proxy_accuracy": best.get("test_baseline_accuracy"),
            "improvement_over_baseline": best.get("test_improvement_over_baseline"),
        }), handle, indent=2)
    return results, best


def _local_density_counts(records):
    """Count proxy-nearest facades by building for local storefront tuning."""
    counts = {}
    totals = {}
    for record in records:
        building_id = record["building_id"]
        proxy_edge = record["proxy_facade_edge_id"]
        totals[building_id] = totals.get(building_id, 0) + 1
        counts.setdefault(building_id, {})
        counts[building_id][proxy_edge] = counts[building_id].get(proxy_edge, 0) + 1
    return counts, totals


def _local_consistency_score(poi_distance, proxy_distance):
    """Score candidates where POI and proxy point support the same facade."""
    gap = abs(float(poi_distance) - float(proxy_distance))
    return distance_score(poi_distance) * distance_score(proxy_distance) * (1.0 / (1.0 + gap))


def _select_local(record, tuned_params, local_params, shared_counts, density_counts, density_totals):
    """Select a facade with local density and POI/proxy consistency bonuses."""
    tuned_selected = _select(record["candidates"], record["nearest_edge_id"], shared_counts, tuned_params)
    building_id = record["building_id"]
    total = density_totals.get(building_id, 0)
    best = None
    best_score = None
    for candidate in record["candidates"]:
        edge_id = candidate["edge_id"]
        poi_distance = float(candidate["distance_m"])
        proxy_distance = float(record["proxy_distances"].get(edge_id, 0.0))
        density = density_counts.get(building_id, {}).get(edge_id, 0) / total if total else 0.0
        consistency = _local_consistency_score(poi_distance, proxy_distance)
        score = (
            tuned_params["distance_weight"] * distance_score(poi_distance)
            + tuned_params["street_alignment_weight"] * candidate["street_alignment_score"]
            + local_params["building_side_density_weight"] * density
            + local_params["poi_proxy_consistency_weight"] * consistency
            - tuned_params["corner_penalty_weight"] * candidate["corner_penalty"]
            - tuned_params["shared_facade_penalty_weight"] * 0.0
        )
        if best_score is None or score > best_score:
            best_score = score
            best = edge_id

    # Guardrail: local evidence must beat tuned selection by a clear margin.
    if best != tuned_selected:
        tuned_candidate = next(candidate for candidate in record["candidates"] if candidate["edge_id"] == tuned_selected)
        tuned_proxy_distance = float(record["proxy_distances"].get(tuned_selected, 0.0))
        tuned_density = density_counts.get(building_id, {}).get(tuned_selected, 0) / total if total else 0.0
        tuned_score = (
            tuned_params["distance_weight"] * distance_score(tuned_candidate["distance_m"])
            + tuned_params["street_alignment_weight"] * tuned_candidate["street_alignment_score"]
            + local_params["building_side_density_weight"] * tuned_density
            + local_params["poi_proxy_consistency_weight"] * _local_consistency_score(tuned_candidate["distance_m"], tuned_proxy_distance)
            - tuned_params["corner_penalty_weight"] * tuned_candidate["corner_penalty"]
        )
        if best_score < tuned_score + 0.10:
            return tuned_selected
    return best


def _evaluate_local(records, tuned_params, local_params, shared_counts, density_counts, density_totals):
    """Evaluate local storefront parameters."""
    tuned_correct = 0
    local_correct = 0
    baseline_correct = 0
    improved = 0
    worsened = 0
    unchanged = 0
    for record in records:
        tuned = _select(record["candidates"], record["nearest_edge_id"], shared_counts, tuned_params)
        local = _select_local(record, tuned_params, local_params, shared_counts, density_counts, density_totals)
        proxy = record["proxy_facade_edge_id"]
        tuned_agree = tuned == proxy
        local_agree = local == proxy
        baseline_agree = record["nearest_edge_id"] == proxy
        tuned_correct += int(tuned_agree)
        local_correct += int(local_agree)
        baseline_correct += int(baseline_agree)
        improved += int(local_agree and not tuned_agree)
        worsened += int(tuned_agree and not local_agree)
        unchanged += int(local_agree == tuned_agree)
    total = len(records)
    return {
        **local_params,
        "row_count": total,
        "baseline_accuracy": round(baseline_correct / total, 6) if total else 0.0,
        "tuned_accuracy_without_local_heuristics": round(tuned_correct / total, 6) if total else 0.0,
        "local_storefront_accuracy": round(local_correct / total, 6) if total else 0.0,
        "improvement_vs_tuned": round((local_correct - tuned_correct) / total, 6) if total else 0.0,
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
    }


def _local_storefront_train_test_tuning(records, tuned_params, shared_counts):
    """Tune local storefront weights on train rows and report held-out test."""
    train_records, test_records = _train_test_split(records)
    density_counts, density_totals = _local_density_counts(train_records)
    test_density_counts, test_density_totals = _local_density_counts(test_records)
    rows = []
    for density_weight, consistency_weight, corner_distance in itertools.product(
        [0.0, 0.05, 0.10, 0.15],
        [0.0, 0.05, 0.10, 0.15],
        [2.0, 3.0, 5.0],
    ):
        params = {
            "building_side_density_weight": density_weight,
            "poi_proxy_consistency_weight": consistency_weight,
            "corner_ambiguity_distance_m": corner_distance,
        }
        train_result = _evaluate_local(train_records, tuned_params, params, shared_counts, density_counts, density_totals)
        test_result = _evaluate_local(test_records, tuned_params, params, shared_counts, test_density_counts, test_density_totals)
        rows.append(
            {
                **params,
                "train_rows": train_result["row_count"],
                "test_rows": test_result["row_count"],
                "train_accuracy": train_result["local_storefront_accuracy"],
                "train_tuned_accuracy": train_result["tuned_accuracy_without_local_heuristics"],
                "test_accuracy": test_result["local_storefront_accuracy"],
                "test_tuned_accuracy": test_result["tuned_accuracy_without_local_heuristics"],
                "test_baseline_accuracy": test_result["baseline_accuracy"],
                "test_improvement_vs_tuned": test_result["improvement_vs_tuned"],
                "test_improved": test_result["improved"],
                "test_worsened": test_result["worsened"],
                "test_unchanged": test_result["unchanged"],
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["train_accuracy", "test_improvement_vs_tuned", "test_worsened"],
        ascending=[False, False, True],
    )
    results.to_csv(LOCAL_STOREFRONT_TUNING_RESULTS_OUTPUT, index=False)
    best = results.iloc[0].to_dict() if not results.empty else {}
    with open(BEST_LOCAL_STOREFRONT_CONFIG_OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)
    return results, best


def grid_search():
    """Run full-grid and train/test tuning, then write analysis outputs."""
    records, context, shared_counts = _build_records()
    if not context["street_available"]:
        raise RuntimeError("Street geometry is required for reranker tuning.")

    results = [_evaluate_params(records, params, context, shared_counts) for params in _iter_params()]
    results_df = pd.DataFrame(results).sort_values(
        ["selected_reranked_proxy_accuracy", "improvement_over_baseline", "worsened"],
        ascending=[False, False, True],
    )
    RERANKER_TUNING_RESULTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RERANKER_TUNING_RESULTS_OUTPUT, index=False)

    best = _best_from_results(results_df)
    best_config = _config_from_result(best)
    with open(BEST_RERANKER_CONFIG_OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(best_config, handle, indent=2)

    selected_rows = _selected_rows(records, best_config, shared_counts)
    accuracy_by_label = _write_accuracy_by_label(selected_rows)
    shared_summary = _write_shared_analysis(selected_rows)
    clean_metrics = _write_clean_subset_outputs(selected_rows, context)
    train_test_results, train_test_best = _train_test_tuning(records, context, shared_counts)
    local_tuning_results, local_tuning_best = _local_storefront_train_test_tuning(records, best_config, shared_counts)

    print(f"Wrote reranker tuning results: {RERANKER_TUNING_RESULTS_OUTPUT} ({len(results_df)} combinations)")
    print(f"Wrote best reranker config: {BEST_RERANKER_CONFIG_OUTPUT}")
    print(f"Wrote label accuracy breakdown: {PROXY_ACCURACY_BY_LABEL_OUTPUT}")
    print(f"Wrote shared-building analysis: {PROXY_SHARED_BUILDING_ANALYSIS_OUTPUT}")
    print(f"Wrote clean subset metrics: {PROXY_CLEAN_SUBSET_METRICS_OUTPUT}")
    print(f"Wrote train/test tuning results: {RERANKER_TUNING_TRAIN_TEST_RESULTS_OUTPUT}")
    print(f"Wrote best train/test reranker config: {BEST_RERANKER_CONFIG_TRAIN_TEST_OUTPUT}")
    print(f"Wrote local storefront tuning results: {LOCAL_STOREFRONT_TUNING_RESULTS_OUTPUT}")
    print(f"Wrote best local storefront config: {BEST_LOCAL_STOREFRONT_CONFIG_OUTPUT}")
    print("\nBest full-data tuning result:")
    print(pd.DataFrame([best]).to_string(index=False))
    print("\nBest train/test result:")
    print(pd.DataFrame([train_test_best]).to_string(index=False))
    print("\nBest local storefront train/test result:")
    print(pd.DataFrame([local_tuning_best]).to_string(index=False))
    print("\nAccuracy by label/group:")
    print(accuracy_by_label.to_string(index=False))
    print("\nClean subset metrics:")
    print(clean_metrics.to_string(index=False))
    print("\nShared-building summary:")
    print(shared_summary.to_string(index=False))
    return results_df, best_config, train_test_results, local_tuning_results


if __name__ == "__main__":
    grid_search()
