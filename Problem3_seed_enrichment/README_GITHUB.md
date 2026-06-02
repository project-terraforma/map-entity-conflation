# Problem 3: Imagery-Seeded POI Verification and Enrichment

This pipeline turns street-level imagery observations into reviewable POI
addition candidates. It is intentionally conservative: an observed sign is a
seed, not enough evidence by itself to publish a new POI.

## Workflow

```text
Zephr sign / entrance observations
-> aggregate observations into one seed per business
-> re-check the local Overture place extract
-> optionally join an explicitly licensed external POI evidence feed
-> score evidence and write a review queue
-> emit proposed additions only for externally verified, locally absent POIs
```

The Zephr source CSVs are expected beside the imagery folder:

```text
../Zephr/boulder_louisville_sign_and_entrance_gt/
|-- boulder_snips_gt_adjusted.csv
|-- louisville_snips_gt_adjusted.csv
`-- Media/
```

Rows with a blank snapshot `overture_id` are the default input. Multiple signs
and doors for the same named business are aggregated into one seed. A front
door coordinate is preferred as the representative location when available.

## Run Locally

Local-only mode does not make network calls:

```bash
python Problem3_seed_enrichment/src/run_pipeline.py
```

Optional licensed evidence enrichment:

```bash
python Problem3_seed_enrichment/src/run_pipeline.py --evidence-csv /path/to/licensed_poi_evidence.csv
```

Useful options:

```bash
python Problem3_seed_enrichment/src/run_pipeline.py --help
python Problem3_seed_enrichment/src/run_pipeline.py --include-existing
python Problem3_seed_enrichment/src/run_pipeline.py --source-dir /path/to/boulder_louisville_sign_and_entrance_gt
```

The evidence CSV must contain `name`, `lat`, `lon`, `source_url`, and `license`.
Optional enrichment columns are `provider`, `provider_id`, `address`,
`category`, `business_status`, `website`, and `phone`.

Do not use Google Places content as an Overture enrichment feed. Google Places
policies restrict storing Places content beyond allowed exceptions, and Google
Maps Platform terms restrict creating content based on Google Maps content.
Google can still be useful as a separate human research surface when used in
accordance with its terms, but it is deliberately not wired into this export.

## Run On Unannotated Images

Raw-image mode runs local Apple Vision OCR before the Overture checks:

```bash
python Problem3_seed_enrichment/src/run_pipeline.py \
  --images-dir /path/to/street_level_images
```

For every image, the extractor:

1. runs Apple Vision text recognition locally
2. reads EXIF GPS when present
3. uses descriptive filenames as weak hints only
4. accepts only confident, geolocated seeds
5. writes ambiguous images to a review queue

Camera systems often strip EXIF GPS. In that case, export the capture route or
collection-app metadata as a CSV:

```bash
python Problem3_seed_enrichment/src/run_pipeline.py \
  --images-dir /path/to/street_level_images \
  --capture-manifest Problem3_seed_enrichment/examples/capture_manifest.example.csv
```

Capture-manifest columns:

| Column | Required | Description |
| --- | --- | --- |
| `image_filename` | yes | Filename under `--images-dir`. |
| `latitude` | recommended | Capture or storefront latitude. |
| `longitude` | recommended | Capture or storefront longitude. |
| `source_dataset` | no | Collection batch identifier. |
| `poi_name_override` | no | Human-reviewed business-name correction. |
| `address_hint` | no | Text seen on the storefront or supplied by the capture app. |

`poi_name_override` is useful after reviewing OCR. It is recorded as an
explicit reviewer decision rather than silently replacing the extracted text.

## Outputs

| File | Description |
| --- | --- |
| `imagery_poi_seeds.csv` | One normalized seed per observed business. |
| `imagery_poi_enriched_candidates.csv` | Full local and external evidence table. |
| `imagery_poi_review_queue.csv` | Rows requiring a human decision. |
| `proposed_overture_additions.csv` | Locally absent, externally verified candidates only. |
| `problem3_summary.csv` | Counts by workflow status. |
| `raw_image_extractions.csv` | Full OCR, EXIF, filename-hint, and acceptance audit table. |
| `raw_image_extraction_review_queue.csv` | Raw images withheld until extraction issues are reviewed. |

## Status Meanings

| Status | Meaning |
| --- | --- |
| `already_in_snapshot` | The source row already had an Overture ID. |
| `current_overture_match` | A likely match exists in the current local Overture extract. |
| `possible_existing_overture_match` | A weaker local Overture candidate needs review. |
| `proposed_new_poi` | Locally absent and externally verified; ready for human review. |
| `external_match_needs_review` | External evidence exists but is ambiguous or weak. |
| `external_match_needs_provenance` | External evidence matched but lacks contribution-ready provenance. |
| `no_external_match` | External lookup completed without a plausible candidate. |
| `pending_external_verification` | Local check completed, but no external provider was run. |

## Publishing Boundary

`proposed_overture_additions.csv` is not an automatic import. A reviewer should
confirm the storefront identity, location, source recency, and proposed
attributes before converting a row into an Overture contribution.
