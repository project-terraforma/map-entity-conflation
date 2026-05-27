"""Configuration for Problem 2: POI <-> building facade matching."""
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
PROBLEM2_ROOT = Path(__file__).resolve().parent

RAW_DATA_DIR = PROJ_ROOT / "data" / "raw"
PROBLEM2_RAW_DATA_DIR = PROBLEM2_ROOT / "data" / "raw"
PROBLEM1_RAW_DATA_DIR = PROJ_ROOT / "Conflation_pipeline" / "data" / "raw"
PROBLEM1_OUTPUT_DIR = PROJ_ROOT / "Conflation_pipeline" / "outputs"
OUTPUT_DIR = PROJ_ROOT / "outputs"

PLACES_INPUT_OPTIONS = ["places.parquet", "places.geojson", "places.csv"]
BUILDINGS_INPUT_OPTIONS = ["buildings.parquet", "buildings.geojson", "buildings.csv"]
ADDRESSES_INPUT_OPTIONS = ["addresses.parquet", "addresses.geojson", "addresses.csv"]
ENTRANCES_INPUT_OPTIONS = [
    "entrances.parquet",
    "entrances.geojson",
    "entrances.csv",
    "connectors.parquet",
    "connectors.geojson",
    "connectors.csv",
]
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
PROBLEM1_OUTPUT_OPTIONS = [
    "final_problem1_conflation.csv",
    "problem1_conflation.csv",
    "final_conflation.csv",
]

FACADE_MATCHES_CSV = OUTPUT_DIR / "problem2_facade_matches.csv"
FACADE_MATCHES_GEOJSON = OUTPUT_DIR / "problem2_facade_matches.geojson"
SUMMARY_CSV = OUTPUT_DIR / "problem2_summary.csv"
REVIEW_SAMPLE = OUTPUT_DIR / "manual_review_facade_sample.csv"
DEBUG_SAMPLE_GEOJSON = OUTPUT_DIR / "problem2_debug_sample.geojson"

BUILDING_NEAREST_FALLBACK_THRESHOLD_M = 35.0
NEAREST_FACADE_HIGH_THRESHOLD_M = 8.0
NEAREST_FACADE_MEDIUM_THRESHOLD_M = 25.0
NEAREST_FACADE_LOW_THRESHOLD_M = 75.0
ENTRANCE_PREFERRED_RADIUS_M = 25.0
STREET_SUPPORT_RADIUS_M = 40.0
MULTIPLE_CLOSE_FACADE_DELTA_M = 2.0
INSIDE_BUILDING_CONFIDENCE_PENALTY = 0.7

MANUAL_SAMPLE_SIZE = 50
DEBUG_SAMPLE_SIZE = 20
