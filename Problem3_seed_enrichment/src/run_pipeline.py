"""Run imagery-seeded POI verification and enrichment end to end."""

import argparse

import pandas as pd

from config import (
    ENRICHED_OUTPUT,
    OUTPUT_DIR,
    PROPOSED_ADDITIONS_OUTPUT,
    REVIEW_QUEUE_OUTPUT,
    SEEDS_OUTPUT,
    SUMMARY_OUTPUT,
)
from licensed_evidence_matcher import load_licensed_evidence, match_seeds as match_licensed_evidence
from local_overture_matcher import load_local_places, match_seeds
from raw_image_extractor import extract_images, write_extraction_outputs
from seed_builder import build_seeds, build_seeds_from_raw_extractions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", help="Directory containing the Zephr CSVs and Media folder.")
    parser.add_argument("--images-dir", help="Directory of unannotated street-level images. Enables raw-image OCR mode.")
    parser.add_argument("--capture-manifest", help="Optional raw-image CSV with filename, location, and reviewer overrides.")
    parser.add_argument("--max-images", type=int, help="Limit raw-image OCR to the first N images for testing.")
    parser.add_argument("--evidence-csv", help="Licensed external POI evidence CSV used for verification and enrichment.")
    parser.add_argument("--include-existing", action="store_true", help="Include observations that already had snapshot Overture IDs.")
    return parser.parse_args()


def classify_workflow_status(row):
    """Choose a review-oriented final status from local and external evidence."""
    if row.get("overture_id_snapshot"):
        return "already_in_snapshot"
    if row.get("local_overture_match_status") == "confirmed":
        return "current_overture_match"
    if row.get("local_overture_match_status") == "possible":
        return "possible_existing_overture_match"
    if row.get("external_match_status") == "verified":
        if row.get("external_source_url") and row.get("external_license"):
            return "proposed_new_poi"
        return "external_match_needs_provenance"
    if row.get("external_match_status") == "needs_review":
        return "external_match_needs_review"
    if row.get("external_match_status") == "not_found":
        return "no_external_match"
    return "pending_external_verification"


def add_external_evidence(df, evidence_csv):
    """Attach explicitly licensed evidence or local-only markers."""
    if evidence_csv:
        return match_licensed_evidence(df, load_licensed_evidence(evidence_csv))
    evidence = [{"external_match_status": "not_configured"} for _ in range(len(df))]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(evidence)], axis=1)


def build_proposed_additions(df):
    """Return reviewable additions in a compact import-oriented shape."""
    additions = df[df["workflow_status"] == "proposed_new_poi"].copy()
    columns = [
        "seed_id",
        "poi_name_seed",
        "external_name",
        "external_address",
        "external_lat",
        "external_lon",
        "external_category",
        "external_business_status",
        "external_website",
        "external_phone",
        "external_provider",
        "external_provider_id",
        "external_source_url",
        "external_license",
        "external_score",
        "observation_count",
        "observed_types",
        "image_references",
        "source_files",
        "source_rows",
        "workflow_status",
    ]
    for column in columns:
        if column not in additions.columns:
            additions[column] = ""
    return additions[columns]


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.images_dir:
        extractions = extract_images(args.images_dir, manifest_path=args.capture_manifest, max_images=args.max_images)
        write_extraction_outputs(extractions)
        seeds = build_seeds_from_raw_extractions(extractions)
        source_dir = args.images_dir
        observation_count = len(extractions)
        print(f"Raw images accepted as geolocated seeds: {int(extractions['accepted_as_seed'].sum())}")
        print(f"Raw images requiring extraction review: {int((~extractions['accepted_as_seed']).sum())}")
    else:
        seeds, source_dir, observation_count = build_seeds(args.source_dir, include_existing=args.include_existing)
    if seeds.empty:
        print("No POI seeds passed extraction. Review raw_image_extraction_review_queue.csv or provide a capture manifest.")
        return
    places, local_paths = load_local_places()
    with_local = match_seeds(seeds, places)
    enriched = add_external_evidence(with_local, args.evidence_csv)
    enriched["workflow_status"] = enriched.apply(classify_workflow_status, axis=1)

    review_queue = enriched[~enriched["workflow_status"].isin({"already_in_snapshot", "current_overture_match"})].copy()
    proposed = build_proposed_additions(enriched)
    summary = enriched.groupby("workflow_status", dropna=False).size().reset_index(name="count")

    seeds.to_csv(SEEDS_OUTPUT, index=False)
    enriched.to_csv(ENRICHED_OUTPUT, index=False)
    review_queue.to_csv(REVIEW_QUEUE_OUTPUT, index=False)
    proposed.to_csv(PROPOSED_ADDITIONS_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print(f"Using source directory: {source_dir}")
    print(f"Usable source observations or images: {observation_count}")
    print(f"Aggregated POI seeds: {len(seeds)}")
    print(f"Local Overture places: {len(places)} from {', '.join(local_paths) if local_paths else 'no local extract'}")
    print(f"Licensed evidence feed: {args.evidence_csv or 'not provided'}")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(f"Wrote enriched candidates: {ENRICHED_OUTPUT.resolve()}")
    print(f"Wrote review queue: {REVIEW_QUEUE_OUTPUT.resolve()}")
    print(f"Wrote proposed additions: {PROPOSED_ADDITIONS_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
