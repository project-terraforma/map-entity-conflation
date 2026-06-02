"""Configuration for imagery-seeded POI verification and enrichment."""

from pathlib import Path

PROBLEM3_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROBLEM3_ROOT.parent
WORKSPACE_ROOT = REPO_ROOT.parent

SOURCE_DIR_OPTIONS = [
    WORKSPACE_ROOT / "Zephr" / "boulder_louisville_sign_and_entrance_gt",
    WORKSPACE_ROOT / "zephr" / "boulder_louisville_sign_and_entrance_gt",
    REPO_ROOT / "benchmark_dataset" / "reference_datasets" / "extracted" / "boulder_louisville_sign_and_entrance_gt",
]
SOURCE_FILES = {
    "boulder": "boulder_snips_gt_adjusted.csv",
    "louisville": "louisville_snips_gt_adjusted.csv",
}

LOCAL_OVERTURE_PLACE_OPTIONS = [
    REPO_ROOT / "Problem2_facade_pipeline" / "data" / "proxy_overture" / "proxy_area_places.parquet",
    REPO_ROOT / "Problem2_facade_pipeline" / "data" / "raw" / "places.geojson",
    REPO_ROOT / "Problem2_facade_pipeline" / "data" / "raw" / "places_boulder.geojson",
    REPO_ROOT / "Problem2_facade_pipeline" / "data" / "raw" / "places_louisville.geojson",
]

OUTPUT_DIR = PROBLEM3_ROOT / "outputs"
SEEDS_OUTPUT = OUTPUT_DIR / "imagery_poi_seeds.csv"
ENRICHED_OUTPUT = OUTPUT_DIR / "imagery_poi_enriched_candidates.csv"
REVIEW_QUEUE_OUTPUT = OUTPUT_DIR / "imagery_poi_review_queue.csv"
PROPOSED_ADDITIONS_OUTPUT = OUTPUT_DIR / "proposed_overture_additions.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "problem3_summary.csv"
RAW_EXTRACTIONS_OUTPUT = OUTPUT_DIR / "raw_image_extractions.csv"
RAW_REVIEW_QUEUE_OUTPUT = OUTPUT_DIR / "raw_image_extraction_review_queue.csv"

LOCAL_SEARCH_RADIUS_M = 120.0
LOCAL_CONFIRMED_SCORE = 0.82
LOCAL_CONFIRMED_NAME_SCORE = 0.80
LOCAL_POSSIBLE_SCORE = 0.62
LOCAL_POSSIBLE_NAME_SCORE = 0.60
EXTERNAL_SEARCH_RADIUS_M = 250.0
EXTERNAL_VERIFIED_SCORE = 0.78
EXTERNAL_VERIFIED_NAME_SCORE = 0.70
EXTERNAL_AMBIGUITY_MARGIN = 0.08
