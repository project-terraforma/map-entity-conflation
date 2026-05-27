# Problem 1: POI, Building, Address, and Street Conflation

This pipeline builds a rule-based baseline for linking Overture POIs to addresses, buildings, and street information.

## What Problem 1 Solves

Overture stores places, addresses, buildings, and streets as separate layers. Problem 1 creates an interpretable relationship table that connects:

```text
POI -> matched address -> matched building -> matched street
```

This output is also the required foundation for Problem 2 facade matching.

## Pipeline Structure

```text
Conflation_pipeline/
|-- src/
|   |-- address_matcher.py
|   |-- building_matcher.py
|   |-- building_validator.py
|   |-- config.py
|   |-- data_loader.py
|   |-- labeler.py
|   |-- normalization.py
|   |-- run_pipeline.py
|   |-- street_connector.py
|   `-- summary_writer.py
|-- outputs/
|-- .gitignore
`-- requirements.txt
```

## Data Inputs Needed

The loader first looks in:

```text
Conflation_pipeline/data/raw/
```

For the run that generated the current outputs, the raw files came from the Problem 2 raw-data folder. After cleanup, that folder is:

```text
Problem2_facade_pipeline/data/raw/
```

Actual inputs used:

| Layer | File | Rows |
| --- | --- | ---: |
| Places | `Problem2_facade_pipeline/data/raw/places.geojson` | 4,039 |
| Addresses | `Problem2_facade_pipeline/data/raw/addresses.geojson` | 4,646 |
| Buildings | `Problem2_facade_pipeline/data/raw/buildings.geojson` | 1,286 |
| Street segments | `Problem2_facade_pipeline/data/raw/segments.geojson` | 2,022 |

If required files are missing, the pipeline prints a clear error listing the searched paths.

## How To Run

Install dependencies:

```bash
pip install -r Conflation_pipeline/requirements.txt
```

Run:

```bash
python Conflation_pipeline/src/run_pipeline.py
```

## Output Files Generated

The successful run generated:

| Output | Description |
| --- | --- |
| `Conflation_pipeline/outputs/final_problem1_conflation.csv` | Final POI/address/building/street relationship table. |
| `Conflation_pipeline/outputs/problem1_summary.csv` | Counts by final label, match method, POI class, building status, and address status. |
| `Conflation_pipeline/outputs/problem1_manual_eval_template.csv` | 50-row manual review template. |

## Actual Result Summary

From `Conflation_pipeline/outputs/problem1_summary.csv`:

| Metric | Count |
| --- | ---: |
| Total POIs | 4,039 |
| `same_building_confirmed` | 2,798 |
| `text_and_distance_match` | 699 |
| `needs_review` | 442 |
| `non_building_poi` | 82 |
| `building_validated_candidate` | 18 |

Other generated counts:

| Field | Label | Count |
| --- | --- | ---: |
| `poi_class` | `building_expected_poi` | 3,957 |
| `poi_class` | `non_building_poi` | 82 |
| `building_status` | `building_consistent` | 3,044 |
| `building_status` | `building_conflict` | 964 |
| `building_status` | `building_possible` | 28 |
| `building_status` | `building_unknown` | 3 |
| `address_match_status` | `matched_high` | 3,271 |
| `address_match_status` | `matched_medium` | 471 |
| `address_match_status` | `uncertain` | 297 |

## Label Meanings

| Label | Meaning |
| --- | --- |
| `same_building_confirmed` | POI and matched address are supported by the same building polygon. |
| `building_validated_candidate` | POI/address match has building support, but not the strongest evidence. |
| `text_and_distance_match` | Address text and distance support the match, but building evidence may still need care. |
| `needs_review` | Evidence is weak, conflicting, or uncertain. |
| `non_building_poi` | POI type/name suggests it should not be evaluated as a building storefront. |

## Known Limitations

- This is a rule-based baseline, not a trained model.
- The current run used fallback raw data now kept locally under `Problem2_facade_pipeline/data/raw/`.
- Some rows have building conflicts and are intentionally labeled for review.
- Street names are mostly derived from matched address text; street segment matching is optional support.
- Manual evaluation is still required for accuracy measurement.

## Future Work

- Add more regions and compare regional error patterns.
- Improve handling of shared addresses and multi-tenant buildings.
- Add formal manual review labels.
- Use stronger street and entrance evidence when available.
- Feed only high-confidence, building-based rows into Problem 2.
