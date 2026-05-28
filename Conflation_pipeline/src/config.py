"""Configuration for the standalone Problem 1 conflation pipeline."""

import os
from pathlib import Path

V3_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V3_ROOT.parent
RAW_DATA_DIR = V3_ROOT / "data" / "raw"
USE_PROXY_AREA_OVERTURE_DATA = os.environ.get("USE_PROXY_AREA_OVERTURE_DATA", "").lower() in {"1", "true", "yes"}
PROXY_OVERTURE_DATA_DIR = REPO_ROOT / "Problem2_facade_pipeline" / "data" / "proxy_overture"
RAW_DATA_DIR_OPTIONS = [
    RAW_DATA_DIR,
    REPO_ROOT / "Problem2_facade_pipeline" / "data" / "raw",
    REPO_ROOT / "data" / "raw",
]
if USE_PROXY_AREA_OVERTURE_DATA:
    RAW_DATA_DIR_OPTIONS.insert(0, PROXY_OVERTURE_DATA_DIR)
PROCESSED_DATA_DIR = V3_ROOT / "data" / "processed"
OUTPUT_DIR = V3_ROOT / "outputs"

PLACES_INPUT_OPTIONS = ["places.parquet", "places.geojson", "places.csv"]
ADDRESSES_INPUT_OPTIONS = ["addresses.parquet", "addresses.geojson", "addresses.csv"]
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
    PLACES_INPUT_OPTIONS.insert(0, "proxy_area_places.parquet")
    ADDRESSES_INPUT_OPTIONS.insert(0, "proxy_area_addresses.parquet")
    BUILDINGS_INPUT_OPTIONS.insert(0, "proxy_area_buildings.parquet")
    STREETS_INPUT_OPTIONS.insert(0, "proxy_area_segments.parquet")

FINAL_OUTPUT = OUTPUT_DIR / "final_problem1_conflation.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "problem1_summary.csv"
MANUAL_EVAL_OUTPUT = OUTPUT_DIR / "problem1_manual_eval_template.csv"
MANUAL_EVAL_SAMPLE_SIZE = 50

ADDRESS_DISTANCE_THRESHOLD_M = 100.0
BUILDING_NEAREST_FALLBACK_THRESHOLD_M = 30.0
STREET_NEAREST_THRESHOLD_M = 50.0
HIGH_CONFIDENCE_THRESHOLD = 0.75
MEDIUM_CONFIDENCE_THRESHOLD = 0.45
TEXT_AND_DISTANCE_FINAL_THRESHOLD = 0.70
CLUSTERED_ADDRESS_MIN_POIS = 5

ENABLE_SAMPLING = False
SAMPLE_SIZE = 1000
ENABLE_BOUNDING_BOX = False
BBOX = (-180.0, -90.0, 180.0, 90.0)  # min_lon, min_lat, max_lon, max_lat

FINAL_OUTPUT_COLUMNS = [
    "poi_id",
    "poi_name",
    "poi_lat",
    "poi_lon",
    "poi_address_input",
    "poi_types",
    "matched_address_id",
    "matched_address_text",
    "matched_address_lat",
    "matched_address_lon",
    "matched_building_id",
    "matched_street",
    "nearest_street_segment_id",
    "nearest_street_name",
    "street_segment_distance_m",
    "street_segment_match_status",
    "distance_m",
    "distance_score",
    "address_similarity",
    "street_similarity",
    "confidence",
    "candidate_count",
    "address_match_status",
    "building_status",
    "real_building_relation",
    "building_validation_label",
    "poi_class",
    "match_method",
    "street_connection_status",
    "final_label",
    "review_reason",
]
