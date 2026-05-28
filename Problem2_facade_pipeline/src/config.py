"""Configuration for Problem 2 facade matching built on Problem 1 outputs."""

import os
from pathlib import Path

PROBLEM2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROBLEM2_ROOT.parent

PROBLEM1_OUTPUT = REPO_ROOT / "Conflation_pipeline" / "outputs" / "final_problem1_conflation.csv"
PROBLEM1_SUMMARY = REPO_ROOT / "Conflation_pipeline" / "outputs" / "problem1_summary.csv"

USE_PROXY_AREA_OVERTURE_DATA = os.environ.get("USE_PROXY_AREA_OVERTURE_DATA", "").lower() in {"1", "true", "yes"}
PROXY_EVAL_INCLUDE_NEEDS_REVIEW = os.environ.get("PROXY_EVAL_INCLUDE_NEEDS_REVIEW", "").lower() in {"1", "true", "yes"}
TUNE_FACADE_RERANKER = os.environ.get("TUNE_FACADE_RERANKER", "").lower() in {"1", "true", "yes"}
ENABLE_SHARED_BUILDING_FACADE_LOGIC = os.environ.get("ENABLE_SHARED_BUILDING_FACADE_LOGIC", "").lower() in {"1", "true", "yes"}
EXCLUDE_NON_BUILDING_POIS_FROM_PROXY_EVAL = os.environ.get(
    "EXCLUDE_NON_BUILDING_POIS_FROM_PROXY_EVAL",
    "1",
).lower() in {"1", "true", "yes"}
PROXY_OVERTURE_DATA_DIR = PROBLEM2_ROOT / "data" / "proxy_overture"
PROXY_BBOX_BUFFER_DEGREES = 0.001
PROXY_BBOX_BUFFER_METERS = None
PROXY_AREA_PLACES_PATH = PROXY_OVERTURE_DATA_DIR / "proxy_area_places.parquet"
PROXY_AREA_BUILDINGS_PATH = PROXY_OVERTURE_DATA_DIR / "proxy_area_buildings.parquet"
PROXY_AREA_ADDRESSES_PATH = PROXY_OVERTURE_DATA_DIR / "proxy_area_addresses.parquet"
PROXY_AREA_STREETS_PATH = PROXY_OVERTURE_DATA_DIR / "proxy_area_segments.parquet"
PROXY_AREA_MANIFEST_PATH = PROXY_OVERTURE_DATA_DIR / "proxy_area_manifest.csv"

RAW_DATA_DIR_OPTIONS = [
    PROBLEM2_ROOT / "data" / "raw",
    REPO_ROOT / "Conflation_pipeline" / "data" / "raw",
    REPO_ROOT / "data" / "raw",
]
if USE_PROXY_AREA_OVERTURE_DATA:
    RAW_DATA_DIR_OPTIONS.insert(0, PROXY_OVERTURE_DATA_DIR)

BUILDINGS_INPUT_OPTIONS = ["buildings.parquet", "buildings.geojson", "buildings.csv"]
STREETS_INPUT_OPTIONS = [
    "streets.parquet",
    "streets.geojson",
    "streets.csv",
    "segments.parquet",
    "segments.geojson",
    "segments.csv",
    "street_segments.parquet",
    "street_segments.geojson",
    "street_segments.csv",
    "streets/segments.parquet",
    "streets/segments.geojson",
    "streets/segments.csv",
]
if USE_PROXY_AREA_OVERTURE_DATA:
    BUILDINGS_INPUT_OPTIONS.insert(0, "proxy_area_buildings.parquet")
    STREETS_INPUT_OPTIONS.insert(0, "proxy_area_segments.parquet")

OUTPUT_DIR = PROBLEM2_ROOT / "outputs"
MATCHES_OUTPUT = OUTPUT_DIR / "problem2_facade_matches.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "problem2_summary.csv"
PROXY_EVALUATION_OUTPUT = OUTPUT_DIR / "problem2_proxy_evaluation.csv"
PROXY_SUMMARY_OUTPUT = OUTPUT_DIR / "problem2_proxy_summary.csv"
PROXY_RANKER_COMPARISON_OUTPUT = OUTPUT_DIR / "problem2_proxy_ranker_comparison.csv"
RERANKER_TUNING_RESULTS_OUTPUT = OUTPUT_DIR / "problem2_reranker_tuning_results.csv"
BEST_RERANKER_CONFIG_OUTPUT = OUTPUT_DIR / "problem2_best_reranker_config.json"
PROXY_ACCURACY_BY_LABEL_OUTPUT = OUTPUT_DIR / "problem2_proxy_accuracy_by_label.csv"
PROXY_SHARED_BUILDING_ANALYSIS_OUTPUT = OUTPUT_DIR / "problem2_proxy_shared_building_analysis.csv"
PROXY_CLEAN_SUBSET_METRICS_OUTPUT = OUTPUT_DIR / "problem2_proxy_clean_subset_metrics.csv"
PROXY_CLEAN_SUBSET_ROWS_OUTPUT = OUTPUT_DIR / "problem2_proxy_clean_subset_rows.csv"
PROXY_CLEAN_SUBSET_EXCLUSIONS_OUTPUT = OUTPUT_DIR / "problem2_proxy_clean_subset_exclusions.csv"
RERANKER_TUNING_TRAIN_TEST_RESULTS_OUTPUT = OUTPUT_DIR / "problem2_reranker_tuning_train_test_results.csv"
BEST_RERANKER_CONFIG_TRAIN_TEST_OUTPUT = OUTPUT_DIR / "problem2_best_reranker_config_train_test.json"

BENCHMARK_DIR = PROBLEM2_ROOT / "benchmark_dataset"
PROXY_CANDIDATES_OUTPUT = BENCHMARK_DIR / "problem2_proxy_candidates.csv"
PROXY_MATCHED_ROWS_OUTPUT = BENCHMARK_DIR / "problem2_proxy_matched_rows.csv"
PROXY_UNMATCHED_ROWS_OUTPUT = BENCHMARK_DIR / "problem2_proxy_unmatched_rows.csv"
PROXY_MATCHED_NOT_EVALUATED_OUTPUT = BENCHMARK_DIR / "problem2_proxy_matched_not_evaluated_rows.csv"
PROXY_RERANKER_IMPROVED_ROWS_OUTPUT = BENCHMARK_DIR / "problem2_proxy_reranker_improved_rows.csv"
PROXY_BASELINE_CORRECT_RERANKER_WRONG_ROWS_OUTPUT = BENCHMARK_DIR / "problem2_proxy_baseline_correct_reranker_wrong_rows.csv"
PROXY_NEEDS_REVIEW_EVALUATED_ROWS_OUTPUT = BENCHMARK_DIR / "problem2_proxy_needs_review_evaluated_rows.csv"
PROXY_CONFIRMED_EVALUATED_ROWS_OUTPUT = BENCHMARK_DIR / "problem2_proxy_confirmed_evaluated_rows.csv"
PROXY_RERANKER_CORRECT_BASELINE_WRONG_OUTPUT = BENCHMARK_DIR / "problem2_proxy_reranker_correct_baseline_wrong.csv"
PROXY_BASELINE_CORRECT_RERANKER_WRONG_OUTPUT = BENCHMARK_DIR / "problem2_proxy_baseline_correct_reranker_wrong.csv"
PROXY_NON_BUILDING_EXCLUDED_OUTPUT = BENCHMARK_DIR / "problem2_proxy_non_building_pois_excluded.csv"
PROXY_EVALUATED_BUILDING_BASED_ROWS_OUTPUT = BENCHMARK_DIR / "problem2_proxy_evaluated_building_based_rows.csv"

PROXY_SOURCE_DIR_OPTIONS = [
    REPO_ROOT / "benchmark_dataset" / "reference_datasets" / "extracted" / "boulder_louisville_sign_and_entrance_gt",
    PROBLEM2_ROOT / "benchmark_dataset" / "reference_datasets" / "extracted" / "boulder_louisville_sign_and_entrance_gt",
]
PROXY_SOURCE_FILES = {
    "boulder": "boulder_snips_gt_adjusted.csv",
    "louisville": "louisville_snips_gt_adjusted.csv",
}
PROXY_NAME_MATCH_MAX_DISTANCE_M = 35.0

VALID_PROBLEM1_FINAL_LABELS = {"same_building_confirmed", "building_validated_candidate"}
NOT_APPLICABLE_FINAL_LABELS = {"non_building_poi"}
NEEDS_REVIEW_FINAL_LABELS = {
    "needs_review",
    "missing_poi_address_no_candidate",
    "spatial_address_candidate",
    "text_and_distance_match",
}
NEEDS_REVIEW_BUILDING_LABELS = {"needs_review", "low_confidence", "clustered_commercial_area"}

NON_BUILDING_PROXY_FINAL_LABELS = {"non_building_poi", "missing_poi_address_no_candidate"}
NON_BUILDING_PROXY_KEYWORDS = {
    "park",
    "trail",
    "trailhead",
    "open space",
    "outdoor",
    "recreation area",
    "natural",
    "lake",
    "reservoir",
    "river",
    "creek",
    "mountain",
    "peak",
    "campground",
    "viewpoint",
    "overlook",
    "transit stop",
    "bus stop",
    "rail station",
    "plaza",
    "square",
}

CORNER_THRESHOLD_M = 4.0
STREET_SEARCH_RADIUS_M = 60.0
STREET_ALIGNMENT_GOOD_DEGREES = 25.0
LONG_EDGE_REFERENCE_M = 30.0
SHARED_FACADE_PENALTY_THRESHOLD = 5
NEAREST_FACADE_LOCKIN_DISTANCE_METERS = float(os.environ.get("NEAREST_FACADE_LOCKIN_DISTANCE_METERS", "5.0"))
MAX_DISTANCE_RATIO_FOR_RERANK_OVERRIDE = float(os.environ.get("MAX_DISTANCE_RATIO_FOR_RERANK_OVERRIDE", "2.0"))
DISTANCE_SCORE_WEIGHT = float(os.environ.get("DISTANCE_SCORE_WEIGHT", "0.75"))
STREET_ALIGNMENT_BONUS_WEIGHT = float(os.environ.get("STREET_ALIGNMENT_BONUS_WEIGHT", "0.10"))
STREET_FACING_BONUS_WEIGHT = float(os.environ.get("STREET_FACING_BONUS_WEIGHT", "0.08"))
EDGE_LENGTH_BONUS_WEIGHT = float(os.environ.get("EDGE_LENGTH_BONUS_WEIGHT", "0.04"))
CORNER_PENALTY_WEIGHT = float(os.environ.get("CORNER_PENALTY_WEIGHT", "0.20"))
SHARED_FACADE_PENALTY_WEIGHT = float(os.environ.get("SHARED_FACADE_PENALTY_WEIGHT", "0.10"))
