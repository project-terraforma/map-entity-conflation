"""Run Problem 2 facade matching as an extension of Problem 1."""

import json
import math
from pathlib import Path

import pandas as pd
from shapely import wkb, wkt
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover
    raise ImportError("Problem 2 requires geopandas. Install Problem2_facade_pipeline/requirements.txt.") from exc

from config import (
    BUILDINGS_INPUT_OPTIONS,
    CORNER_THRESHOLD_M,
    LONG_EDGE_REFERENCE_M,
    MATCHES_OUTPUT,
    BEST_RERANKER_CONFIG_OUTPUT,
    CORNER_PENALTY_WEIGHT,
    DISTANCE_SCORE_WEIGHT,
    EDGE_LENGTH_BONUS_WEIGHT,
    ENABLE_SHARED_BUILDING_FACADE_LOGIC,
    FACADE_SEGMENT_LENGTH_METERS,
    MAX_DISTANCE_RATIO_FOR_RERANK_OVERRIDE,
    NEEDS_REVIEW_BUILDING_LABELS,
    NEEDS_REVIEW_FINAL_LABELS,
    NEAREST_FACADE_LOCKIN_DISTANCE_METERS,
    NOT_APPLICABLE_FINAL_LABELS,
    OUTPUT_DIR,
    PROBLEM1_OUTPUT,
    PROXY_EVAL_INCLUDE_NEEDS_REVIEW,
    RAW_DATA_DIR_OPTIONS,
    MALL_OR_PLAZA_POI_THRESHOLD,
    SHARED_BUILDING_DISTANCE_WEIGHT,
    SHARED_BUILDING_MIN_POIS,
    SHARED_FACADE_PENALTY_THRESHOLD,
    SHARED_FACADE_PENALTY_WEIGHT,
    SHARED_BUILDING_STREET_WEIGHT,
    SPLIT_LONG_FACADES_FOR_SHARED_BUILDINGS,
    STREET_ALIGNMENT_GOOD_DEGREES,
    STREET_ALIGNMENT_BONUS_WEIGHT,
    STREET_FACING_BONUS_WEIGHT,
    STREET_SEARCH_RADIUS_M,
    STREETS_INPUT_OPTIONS,
    SUMMARY_OUTPUT,
    TUNE_FACADE_RERANKER,
    VALID_PROBLEM1_FINAL_LABELS,
)

PROBLEM2_EVAL_MODE = "proxy_permissive" if PROXY_EVAL_INCLUDE_NEEDS_REVIEW else "strict"
MALL_OR_PLAZA_KEYWORDS = {
    "mall",
    "plaza",
    "shopping center",
    "center",
    "outlet",
    "market",
    "square",
    "suite",
    "unit",
    "store",
    "shops",
    "village",
    "commons",
    "marketplace",
}


def load_tuned_reranker_config():
    """Load optional tuned reranker weights without changing default behavior."""
    config = {
        "distance_weight": DISTANCE_SCORE_WEIGHT,
        "street_alignment_weight": STREET_ALIGNMENT_BONUS_WEIGHT,
        "street_facing_weight": STREET_FACING_BONUS_WEIGHT,
        "edge_length_weight": EDGE_LENGTH_BONUS_WEIGHT,
        "corner_penalty_weight": CORNER_PENALTY_WEIGHT,
        "shared_facade_penalty_weight": SHARED_FACADE_PENALTY_WEIGHT,
        "lockin_distance_m": NEAREST_FACADE_LOCKIN_DISTANCE_METERS,
        "max_distance_ratio_for_override": MAX_DISTANCE_RATIO_FOR_RERANK_OVERRIDE,
        "strong_override_margin": 0.15,
    }
    if TUNE_FACADE_RERANKER and BEST_RERANKER_CONFIG_OUTPUT.exists():
        with open(BEST_RERANKER_CONFIG_OUTPUT, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        config.update({key: loaded[key] for key in config if key in loaded})
    return config


def is_missing(value):
    """Return True for empty CSV values."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip() == ""


def parse_geometry(value):
    """Parse shapely, WKT, or WKB geometry."""
    if is_missing(value):
        return None
    if hasattr(value, "geom_type"):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return wkb.loads(bytes(value))
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.upper().startswith(("POINT", "LINESTRING", "POLYGON", "MULTI")):
                return wkt.loads(text)
            return wkb.loads(bytes.fromhex(text))
        except Exception:
            return None
    return None


def find_input_file(options, layer_name, required=True):
    """Find an input file across known raw-data locations."""
    searched = []
    for raw_dir in RAW_DATA_DIR_OPTIONS:
        for option in options:
            path = raw_dir / option
            searched.append(path)
            if path.exists():
                return path
    if required:
        searched_text = "\n".join(f"- {path}" for path in searched)
        raise FileNotFoundError(f"No {layer_name} file found. Looked in:\n{searched_text}")
    return None


def load_table(path: Path):
    """Load CSV, GeoJSON, or Parquet."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".geojson", ".json"}:
        return gpd.read_file(path)
    raise ValueError(f"Unsupported input format: {path}")


def first_existing(df, names):
    """Return the first matching column name."""
    lower = {str(col).lower(): col for col in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def load_problem1_output():
    """Load the freshly generated Problem 1 output."""
    if not PROBLEM1_OUTPUT.exists():
        raise FileNotFoundError(
            f"Problem 1 output is missing: {PROBLEM1_OUTPUT}. "
            "Run python Conflation_pipeline/src/run_pipeline.py first."
        )
    df = pd.read_csv(PROBLEM1_OUTPUT, low_memory=False)
    required = {"poi_id", "poi_lat", "poi_lon", "matched_building_id", "final_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Problem 1 output is missing required columns: {sorted(missing)}")
    return df


def load_buildings():
    """Load building polygons keyed by building id."""
    path = find_input_file(BUILDINGS_INPUT_OPTIONS, "buildings", required=True)
    df = load_table(path)
    id_col = first_existing(df, ["building_id", "id", "overture_id"])
    geom_col = first_existing(df, ["geometry", "geom", "wkb_geometry", "geometry_wkt"])
    if id_col is None or geom_col is None:
        raise ValueError(f"Building file must contain id and geometry columns: {path}")
    out = df[[id_col, geom_col]].copy().rename(columns={id_col: "building_id", geom_col: "geometry"})
    out["building_id"] = out["building_id"].astype(str)
    out["geometry"] = out["geometry"].apply(parse_geometry)
    out = out[out["geometry"].apply(lambda geom: geom is not None and not geom.is_empty)].copy()
    return gpd.GeoDataFrame(out, geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326"), path


def load_streets(metric_crs):
    """Load optional street geometries projected to the metric CRS."""
    path = find_input_file(STREETS_INPUT_OPTIONS, "streets/segments", required=False)
    if path is None:
        return None, None
    df = load_table(path)
    geom_col = first_existing(df, ["geometry", "geom", "wkb_geometry", "geometry_wkt"])
    if geom_col is None:
        return None, path
    out = df[[geom_col]].copy().rename(columns={geom_col: "geometry"})
    out["geometry"] = out["geometry"].apply(parse_geometry)
    out = out[out["geometry"].apply(lambda geom: geom is not None and not geom.is_empty)].copy()
    out = out[out["geometry"].apply(lambda geom: geom.geom_type in {"LineString", "MultiLineString"})].copy()
    if out.empty:
        return None, path
    return gpd.GeoDataFrame(out, geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326").to_crs(metric_crs), path


def classify_problem2_row(row):
    """Classify whether a Problem 1 row should continue to facade matching."""
    final_label = str(row.get("final_label", "") or "")
    poi_class = str(row.get("poi_class", "") or "")
    building_id = row.get("matched_building_id")
    building_validation = str(row.get("building_validation_label", "") or "")

    if final_label in NOT_APPLICABLE_FINAL_LABELS or poi_class == "non_building_poi":
        return "not_applicable"
    if is_missing(building_id):
        return "no_building_match"
    if final_label in VALID_PROBLEM1_FINAL_LABELS and building_validation not in NEEDS_REVIEW_BUILDING_LABELS:
        return "facade_candidate"
    if PROXY_EVAL_INCLUDE_NEEDS_REVIEW and final_label == "needs_review":
        return "facade_candidate"
    if final_label in NEEDS_REVIEW_FINAL_LABELS or building_validation in NEEDS_REVIEW_BUILDING_LABELS:
        return "needs_review"
    return "needs_review"


def proxy_eval_included_needs_review(row):
    """Return True when a row was admitted only by permissive proxy evaluation mode."""
    return bool(
        PROXY_EVAL_INCLUDE_NEEDS_REVIEW
        and str(row.get("final_label", "") or "") == "needs_review"
        and row.get("problem2_status") == "facade_candidate"
    )


def polygon_parts(geometry):
    """Return polygon parts for Polygon or MultiPolygon."""
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    return []


def line_bearing(line):
    """Return a compass-like bearing in degrees."""
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    return (math.degrees(math.atan2(x2 - x1, y2 - y1)) + 360.0) % 360.0


def angle_difference(a, b):
    """Smallest angle difference for unoriented line alignment."""
    if a is None or b is None:
        return None
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return min(diff, 180.0 - diff)


def extract_facades(building_id, geometry):
    """Split building exteriors into individual facade edge records."""
    facades = []
    index = 0
    for polygon in polygon_parts(geometry):
        coords = list(polygon.exterior.coords)
        for i in range(len(coords) - 1):
            line = LineString([coords[i], coords[i + 1]])
            if line.length <= 0:
                continue
            facades.append(
                {
                    "edge_id": f"{building_id}:{index}",
                    "geometry": line,
                    "length_m": float(line.length),
                    "bearing": line_bearing(line),
                }
            )
            index += 1
    return facades


def split_facade_segments(facades):
    """Split long facade edges into smaller segment candidates for experiments."""
    segments = []
    for facade in facades:
        line = facade["geometry"]
        segment_count = max(1, math.ceil(float(facade["length_m"]) / FACADE_SEGMENT_LENGTH_METERS))
        if segment_count == 1:
            item = dict(facade)
            item["segment_id"] = item["edge_id"]
            item["parent_facade_id"] = item["edge_id"]
            item["segment_index"] = 0
            item["segment_length_m"] = item["length_m"]
            segments.append(item)
            continue
        for idx in range(segment_count):
            start = line.interpolate(idx / segment_count, normalized=True)
            end = line.interpolate((idx + 1) / segment_count, normalized=True)
            segment = LineString([start, end])
            edge_id = f"{facade['edge_id']}:seg{idx}"
            segments.append(
                {
                    "edge_id": edge_id,
                    "geometry": segment,
                    "length_m": float(segment.length),
                    "bearing": line_bearing(segment),
                    "segment_id": edge_id,
                    "parent_facade_id": facade["edge_id"],
                    "segment_index": idx,
                    "segment_length_m": float(segment.length),
                }
            )
    return segments


def nearest_street_info(streets_gdf, edge_geom):
    """Return nearest street distance and bearing for a facade edge."""
    if streets_gdf is None or streets_gdf.empty:
        return None, None
    try:
        search = edge_geom.buffer(STREET_SEARCH_RADIUS_M)
        idx = list(streets_gdf.sindex.query(search, predicate="intersects"))
        candidates = streets_gdf.iloc[idx] if idx else streets_gdf
        distances = candidates.geometry.distance(edge_geom)
        nearest_idx = distances.idxmin()
        nearest_geom = candidates.loc[nearest_idx].geometry
        return float(distances.loc[nearest_idx]), line_bearing(nearest_geom)
    except Exception:
        return None, None


def distance_score(distance_m):
    """Convert POI-edge distance into a 0..1 score."""
    return 1.0 / (1.0 + max(0.0, float(distance_m)))


def street_facing_score(street_distance_m):
    """Score edge closeness to street geometry."""
    if street_distance_m is None or pd.isna(street_distance_m):
        return 0.0
    return 1.0 / (1.0 + max(0.0, float(street_distance_m)) / STREET_SEARCH_RADIUS_M)


def street_alignment_score(edge_bearing, street_bearing):
    """Score parallel alignment between facade and nearby street."""
    diff = angle_difference(edge_bearing, street_bearing)
    if diff is None:
        return 0.0
    return max(0.0, 1.0 - diff / 90.0)


def edge_candidate(row, poi_point, facade, streets_gdf):
    """Compute all baseline geometry fields for one facade candidate."""
    edge_geom = facade["geometry"]
    projected = nearest_points(poi_point, edge_geom)[1]
    coords = list(edge_geom.coords)
    endpoint_distance = min(projected.distance(Point(coords[0])), projected.distance(Point(coords[-1])))
    corner_penalty = 0.25 if endpoint_distance <= CORNER_THRESHOLD_M else 0.0
    street_distance, street_bearing = nearest_street_info(streets_gdf, edge_geom)
    align_score = street_alignment_score(facade["bearing"], street_bearing)
    facing_score = street_facing_score(street_distance)
    length_score = min(1.0, facade["length_m"] / LONG_EDGE_REFERENCE_M)

    return {
        "poi_id": row.get("poi_id"),
        "edge_id": facade["edge_id"],
        "edge_geometry": edge_geom,
        "distance_m": float(poi_point.distance(edge_geom)),
        "edge_length_m": float(facade["length_m"]),
        "edge_bearing": facade["bearing"],
        "projected_position_m": float(edge_geom.project(projected)),
        "near_corner": endpoint_distance <= CORNER_THRESHOLD_M,
        "endpoint_distance_m": float(endpoint_distance),
        "street_distance_m": street_distance,
        "street_bearing": street_bearing,
        "street_alignment_score": align_score,
        "street_facing_score": facing_score,
        "edge_length_score": length_score,
        "corner_penalty": corner_penalty,
    }


def score_candidate(candidate, shared_count, street_available):
    """Score a candidate facade edge for street-aware re-ranking."""
    shared_penalty = 0.10 if shared_count >= SHARED_FACADE_PENALTY_THRESHOLD else 0.0
    candidate["shared_facade_penalty"] = shared_penalty
    if not street_available:
        candidate["selected_facade_score"] = distance_score(candidate["distance_m"])
        return candidate

    score = (
        0.45 * distance_score(candidate["distance_m"])
        + 0.25 * candidate["street_facing_score"]
        + 0.15 * candidate["street_alignment_score"]
        + 0.10 * candidate["edge_length_score"]
        - candidate["corner_penalty"]
        - shared_penalty
    )
    candidate["selected_facade_score"] = round(float(score), 6)
    return candidate


def score_candidate_tuned(candidate, shared_count, street_available, reranker_config):
    """Distance-dominant tuned score used only when TUNE_FACADE_RERANKER is enabled."""
    shared_penalty = 0.10 if shared_count >= SHARED_FACADE_PENALTY_THRESHOLD else 0.0
    candidate["shared_facade_penalty"] = shared_penalty
    if not street_available:
        candidate["selected_facade_score"] = distance_score(candidate["distance_m"])
        return candidate

    score = (
        reranker_config["distance_weight"] * distance_score(candidate["distance_m"])
        + reranker_config["street_alignment_weight"] * candidate["street_alignment_score"]
        + reranker_config["street_facing_weight"] * candidate["street_facing_score"]
        + reranker_config["edge_length_weight"] * candidate["edge_length_score"]
        - reranker_config["corner_penalty_weight"] * candidate["corner_penalty"]
        - reranker_config["shared_facade_penalty_weight"] * shared_penalty
    )
    candidate["selected_facade_score"] = round(float(score), 6)
    return candidate


def score_candidate_shared(candidate, shared_count, street_available, reranker_config):
    """Shared-building score: local distance dominates; street is a small bonus."""
    shared_penalty = 0.10 if shared_count >= SHARED_FACADE_PENALTY_THRESHOLD else 0.0
    candidate["shared_facade_penalty"] = shared_penalty
    score = SHARED_BUILDING_DISTANCE_WEIGHT * distance_score(candidate["distance_m"])
    if street_available:
        score += SHARED_BUILDING_STREET_WEIGHT * candidate["street_alignment_score"]
        score += SHARED_BUILDING_STREET_WEIGHT * candidate["street_facing_score"]
    score -= reranker_config["corner_penalty_weight"] * candidate["corner_penalty"]
    score -= reranker_config["shared_facade_penalty_weight"] * shared_penalty
    candidate["selected_facade_score"] = round(float(score), 6)
    return candidate


def choose_selected_candidate(scored, nearest, street_available, reranker_config):
    """Choose a facade candidate, preserving the existing reranker unless tuning is enabled."""
    if not street_available:
        return nearest
    if not TUNE_FACADE_RERANKER:
        return max(scored, key=lambda item: item["selected_facade_score"])

    nearest_distance = max(float(nearest["distance_m"]), 0.001)
    max_ratio = float(reranker_config["max_distance_ratio_for_override"])
    eligible = [
        candidate
        for candidate in scored
        if float(candidate["distance_m"]) <= nearest_distance * max_ratio
    ]
    selected = max(eligible or scored, key=lambda item: item["selected_facade_score"])

    if nearest_distance <= float(reranker_config["lockin_distance_m"]):
        margin = float(reranker_config.get("strong_override_margin", 0.15))
        nearest_score = float(nearest["selected_facade_score"])
        selected_score = float(selected["selected_facade_score"])
        if selected["edge_id"] != nearest["edge_id"] and selected_score < nearest_score + margin:
            return nearest
    return selected


def empty_output_row(row, status):
    """Return a Problem 2 row for records not sent to facade matching."""
    included_needs_review = proxy_eval_included_needs_review(row)
    return {
        "poi_id": row.get("poi_id"),
        "poi_name": row.get("poi_name"),
        "problem1_final_label": row.get("final_label"),
        "source_final_label": row.get("final_label"),
        "problem2_eval_mode": PROBLEM2_EVAL_MODE,
        "proxy_eval_included_needs_review": included_needs_review,
        "problem2_status": status,
        "building_id": row.get("matched_building_id"),
        "nearest_edge_id": None,
        "nearest_edge_distance_m": None,
        "nearest_edge_bearing": None,
        "nearest_edge_length_m": None,
        "selected_facade_edge_id": None,
        "selected_facade_distance_m": None,
        "selected_facade_score": None,
        "selected_method": status,
        "selected_facade_reason": f"Problem 1 classified this row as {status}.",
        "street_used_for_rerank": False,
        "street_alignment_score": None,
        "street_facing_score": None,
        "facade_edge_length_m": None,
        "facade_edge_bearing": None,
        "corner_penalty": None,
        "shared_facade_penalty": None,
        "shared_building_poi_count": None,
        "is_shared_building": False,
        "possible_mall_or_plaza": False,
        "shared_building_mode_applied": False,
        "selected_facade_before_shared_logic": None,
        "selected_facade_after_shared_logic": None,
        "shared_logic_changed_selection": False,
        "shared_logic_improved_proxy_match": None,
        "shared_logic_worsened_proxy_match": None,
        "proxy_distance_before_shared_logic": None,
        "proxy_distance_after_shared_logic": None,
    }


def is_possible_mall_or_plaza(row, shared_building_count):
    """Detect possible mall/plaza/shared-commercial cases using real POI text."""
    if shared_building_count >= MALL_OR_PLAZA_POI_THRESHOLD:
        return True
    text = " ".join(str(row.get(col, "") or "").lower() for col in ["poi_name", "poi_types", "poi_address_input"])
    return any(keyword in text for keyword in MALL_OR_PLAZA_KEYWORDS)


def build_summary(output_df, total_rows):
    """Build the Problem 2 summary CSV."""
    status_counts = output_df["problem2_status"].value_counts().to_dict()
    method_counts = output_df["selected_method"].value_counts().to_dict()
    numeric = output_df[output_df["problem2_status"] == "facade_candidate"]
    included_needs_review_count = int(output_df.get("proxy_eval_included_needs_review", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    rows = [
        {"metric": "total_problem1_rows_consumed", "value": int(total_rows)},
        {"metric": "problem2_eval_mode", "value": PROBLEM2_EVAL_MODE},
        {"metric": "proxy_eval_include_needs_review", "value": bool(PROXY_EVAL_INCLUDE_NEEDS_REVIEW)},
        {"metric": "not_applicable_count", "value": int(status_counts.get("not_applicable", 0))},
        {"metric": "no_building_match_count", "value": int(status_counts.get("no_building_match", 0))},
        {"metric": "needs_review_count", "value": int(status_counts.get("needs_review", 0))},
        {"metric": "proxy_eval_included_needs_review_count", "value": included_needs_review_count},
        {"metric": "facade_candidate_count", "value": int(status_counts.get("facade_candidate", 0))},
        {"metric": "nearest_edge_baseline_count", "value": int(method_counts.get("nearest_edge_baseline", 0))},
        {"metric": "street_aware_rerank_count", "value": int(method_counts.get("street_aware_rerank", 0))},
        {"metric": "average_nearest_edge_distance", "value": round(float(numeric["nearest_edge_distance_m"].mean()), 3) if not numeric.empty else None},
        {"metric": "average_selected_facade_distance", "value": round(float(numeric["selected_facade_distance_m"].mean()), 3) if not numeric.empty else None},
        {"metric": "average_selected_facade_score", "value": round(float(numeric["selected_facade_score"].mean()), 6) if not numeric.empty else None},
    ]
    return pd.DataFrame(rows)


def run():
    """Run Problem 2 from Problem 1 output to facade outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    problem1 = load_problem1_output()
    buildings, building_path = load_buildings()
    buildings_m = buildings.to_crs(buildings.estimate_utm_crs() or "EPSG:3857")
    streets_m, street_path = load_streets(buildings_m.crs)
    street_available = streets_m is not None and not streets_m.empty
    reranker_config = load_tuned_reranker_config()

    print(f"Loaded Problem 1 output: {PROBLEM1_OUTPUT} ({len(problem1)} rows)")
    print(f"Loaded buildings: {building_path} ({len(buildings)} rows)")
    print(f"Loaded streets: {street_path if street_path else 'not provided'} ({len(streets_m) if street_available else 0} rows)")
    print(f"Problem 2 evaluation mode: {PROBLEM2_EVAL_MODE}")
    print(f"Tuned facade reranker enabled: {bool(TUNE_FACADE_RERANKER)}")
    if PROXY_EVAL_INCLUDE_NEEDS_REVIEW:
        print("Permissive proxy evaluation is enabled: final_label=needs_review rows can receive facade candidates for benchmarking only.")

    problem1["problem2_status"] = problem1.apply(classify_problem2_row, axis=1)
    building_lookup = buildings_m.drop_duplicates("building_id").set_index("building_id")
    building_poi_counts = problem1[problem1["problem2_status"] == "facade_candidate"]["matched_building_id"].astype(str).value_counts().to_dict()

    candidate_cache = {}
    output_rows = []
    nearest_edge_ids = []

    # First pass: classify rows and compute nearest-edge baseline choices.
    for _, row in problem1.iterrows():
        status = row["problem2_status"]
        if status != "facade_candidate":
            output_rows.append(empty_output_row(row, status))
            continue

        building_id = str(row.get("matched_building_id"))
        if building_id not in building_lookup.index or is_missing(row.get("poi_lat")) or is_missing(row.get("poi_lon")):
            output_rows.append(empty_output_row(row, "no_building_match"))
            continue

        poi = gpd.GeoDataFrame(
            [{"geometry": Point(float(row["poi_lon"]), float(row["poi_lat"]))}],
            geometry="geometry",
            crs="EPSG:4326",
        ).to_crs(buildings_m.crs).iloc[0].geometry
        shared_building_count = int(building_poi_counts.get(str(building_id), 0))
        is_shared_building = shared_building_count >= SHARED_BUILDING_MIN_POIS
        facades = extract_facades(building_id, building_lookup.loc[building_id].geometry)
        if SPLIT_LONG_FACADES_FOR_SHARED_BUILDINGS and ENABLE_SHARED_BUILDING_FACADE_LOGIC and is_shared_building:
            facades = split_facade_segments(facades)
        candidates = [edge_candidate(row, poi, facade, streets_m) for facade in facades]
        if not candidates:
            output_rows.append(empty_output_row(row, "no_building_match"))
            continue
        nearest = min(candidates, key=lambda item: item["distance_m"])
        nearest_edge_ids.append(nearest["edge_id"])
        candidate_cache[row["poi_id"]] = (row, candidates, nearest)
        output_rows.append({"poi_id": row["poi_id"], "_from_candidate_cache": True})

    shared_counts = pd.Series(nearest_edge_ids).value_counts().to_dict()

    # Second pass: apply street-aware score and write final facade selections.
    cache_by_poi = {key: value for key, value in candidate_cache.items()}
    final_rows = []
    for existing in output_rows:
        poi_id = existing.get("poi_id")
        if poi_id not in cache_by_poi:
            final_rows.append(existing)
            continue

        row, candidates, nearest = cache_by_poi[poi_id]
        local_reranker_config = dict(reranker_config)
        shared_building_count = int(building_poi_counts.get(str(row.get("matched_building_id")), 0))
        is_shared_building = shared_building_count >= SHARED_BUILDING_MIN_POIS
        possible_mall_or_plaza = is_possible_mall_or_plaza(row, shared_building_count)
        shared_building_logic_used = bool(ENABLE_SHARED_BUILDING_FACADE_LOGIC and is_shared_building)
        if TUNE_FACADE_RERANKER and shared_building_logic_used:
            local_reranker_config["street_alignment_weight"] = min(local_reranker_config["street_alignment_weight"], 0.03)
            local_reranker_config["street_facing_weight"] = min(local_reranker_config["street_facing_weight"], 0.03)
            local_reranker_config["distance_weight"] = max(local_reranker_config["distance_weight"], 0.85)
            local_reranker_config["max_distance_ratio_for_override"] = min(local_reranker_config["max_distance_ratio_for_override"], 1.5)
        scorer = score_candidate_tuned if TUNE_FACADE_RERANKER else score_candidate
        if TUNE_FACADE_RERANKER:
            scored = [
                scorer(candidate, shared_counts.get(candidate["edge_id"], 0), street_available, local_reranker_config)
                for candidate in candidates
            ]
        else:
            scored = [scorer(candidate, shared_counts.get(candidate["edge_id"], 0), street_available) for candidate in candidates]
        original_selected = choose_selected_candidate(scored, nearest, street_available, reranker_config)
        selected = original_selected
        if shared_building_logic_used:
            shared_scored = [
                score_candidate_shared(dict(candidate), shared_counts.get(candidate["edge_id"], 0), street_available, local_reranker_config)
                for candidate in candidates
            ]
            selected = choose_selected_candidate(shared_scored, nearest, street_available, local_reranker_config)
        if not street_available:
            selected = score_candidate(selected, shared_counts.get(selected["edge_id"], 0), False)

        method = "tuned_street_aware_rerank" if street_available and TUNE_FACADE_RERANKER else ("street_aware_rerank" if street_available else "nearest_edge_baseline")
        reason_parts = [f"nearest_edge_baseline={nearest['edge_id']}"]
        included_needs_review = proxy_eval_included_needs_review(row)
        if included_needs_review:
            reason_parts.append("proxy evaluation mode included needs_review row; not production-confirmed")
        if TUNE_FACADE_RERANKER:
            reason_parts.append("distance-dominant tuned reranker and guardrails enabled")
        if shared_building_logic_used:
            reason_parts.append("shared-building experimental logic reduced street weights")
        if street_available:
            reason_parts.append("street geometry used for distance/alignment scoring")
        else:
            reason_parts.append("street-aware scoring skipped because no street geometry was available")
        if selected["near_corner"]:
            reason_parts.append("corner penalty applied")
        if selected["shared_facade_penalty"]:
            reason_parts.append("shared facade penalty applied")

        final_rows.append(
            {
                "poi_id": row.get("poi_id"),
                "poi_name": row.get("poi_name"),
                "problem1_final_label": row.get("final_label"),
                "source_final_label": row.get("final_label"),
                "problem2_eval_mode": PROBLEM2_EVAL_MODE,
                "proxy_eval_included_needs_review": included_needs_review,
                "problem2_status": "facade_candidate",
                "building_id": row.get("matched_building_id"),
                "nearest_edge_id": nearest["edge_id"],
                "nearest_edge_distance_m": round(nearest["distance_m"], 3),
                "nearest_edge_bearing": round(nearest["edge_bearing"], 3) if nearest["edge_bearing"] is not None else None,
                "nearest_edge_length_m": round(nearest["edge_length_m"], 3),
                "selected_facade_edge_id": selected["edge_id"],
                "selected_facade_distance_m": round(selected["distance_m"], 3),
                "selected_facade_score": selected["selected_facade_score"],
                "selected_method": method,
                "selected_facade_reason": "; ".join(reason_parts),
                "street_used_for_rerank": bool(street_available),
                "street_alignment_score": round(selected["street_alignment_score"], 6),
                "street_facing_score": round(selected["street_facing_score"], 6),
                "facade_edge_length_m": round(selected["edge_length_m"], 3),
                "facade_edge_bearing": round(selected["edge_bearing"], 3) if selected["edge_bearing"] is not None else None,
                "corner_penalty": selected["corner_penalty"],
                "shared_facade_penalty": selected["shared_facade_penalty"],
                "shared_building_poi_count": shared_building_count,
                "is_shared_building": bool(is_shared_building),
                "possible_mall_or_plaza": bool(possible_mall_or_plaza),
                "shared_building_mode_applied": bool(shared_building_logic_used),
                "selected_facade_before_shared_logic": original_selected["edge_id"],
                "selected_facade_after_shared_logic": selected["edge_id"],
                "shared_logic_changed_selection": bool(original_selected["edge_id"] != selected["edge_id"]),
                "shared_logic_improved_proxy_match": None,
                "shared_logic_worsened_proxy_match": None,
                "proxy_distance_before_shared_logic": None,
                "proxy_distance_after_shared_logic": None,
            }
        )

    output_df = pd.DataFrame(final_rows)
    output_df.to_csv(MATCHES_OUTPUT, index=False)
    summary = build_summary(output_df, len(problem1))
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print(f"Wrote Problem 2 matches: {MATCHES_OUTPUT}")
    print(f"Wrote Problem 2 summary: {SUMMARY_OUTPUT}")
    print("\nProblem 2 Summary:")
    print(summary.to_string(index=False))
    print("\nProblem 2 status counts:")
    print(output_df["problem2_status"].value_counts().to_string())
    print("\nSelected method counts:")
    print(output_df["selected_method"].value_counts().to_string())

    try:
        from proxy_benchmark_builder import build_proxy_candidates
        from proxy_facade_evaluator import run_proxy_evaluation

        print("\nStarting optional Louisville/Boulder entrance/sign proxy evaluation...")
        proxy_candidates = build_proxy_candidates()
        if proxy_candidates is None or proxy_candidates.empty:
            print("Proxy evaluation skipped because no usable Louisville/Boulder proxy rows were available.")
        else:
            run_proxy_evaluation()
    except FileNotFoundError as exc:
        print(f"Proxy evaluation skipped: {exc}")
    except Exception as exc:
        print(f"Proxy evaluation skipped due to an unexpected error: {exc}")


if __name__ == "__main__":
    run()
