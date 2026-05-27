"""Configuration for Problem 2 facade matching built on Problem 1 outputs."""

from pathlib import Path

PROBLEM2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROBLEM2_ROOT.parent

PROBLEM1_OUTPUT = REPO_ROOT / "Conflation_pipeline" / "outputs" / "final_problem1_conflation.csv"
PROBLEM1_SUMMARY = REPO_ROOT / "Conflation_pipeline" / "outputs" / "problem1_summary.csv"

RAW_DATA_DIR_OPTIONS = [
    PROBLEM2_ROOT / "data" / "raw",
    REPO_ROOT / "Conflation_pipeline" / "data" / "raw",
    REPO_ROOT / "data" / "raw",
]

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

OUTPUT_DIR = PROBLEM2_ROOT / "outputs"
MATCHES_OUTPUT = OUTPUT_DIR / "problem2_facade_matches.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "problem2_summary.csv"

VALID_PROBLEM1_FINAL_LABELS = {"same_building_confirmed", "building_validated_candidate"}
NOT_APPLICABLE_FINAL_LABELS = {"non_building_poi"}
NEEDS_REVIEW_FINAL_LABELS = {
    "needs_review",
    "missing_poi_address_no_candidate",
    "spatial_address_candidate",
    "text_and_distance_match",
}
NEEDS_REVIEW_BUILDING_LABELS = {"needs_review", "low_confidence", "clustered_commercial_area"}

CORNER_THRESHOLD_M = 4.0
STREET_SEARCH_RADIUS_M = 60.0
STREET_ALIGNMENT_GOOD_DEGREES = 25.0
LONG_EDGE_REFERENCE_M = 30.0
SHARED_FACADE_PENALTY_THRESHOLD = 5
