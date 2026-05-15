# Overture Entity Conflation Baseline: POIs ↔ Addresses ↔ Buildings ↔ Streets

Gauri Jain, Angad Pantvaidya

## Project Summary

This repository contains a standalone baseline pipeline for **Overture Maps entity conflation, Problem 1**. The pipeline links **Points of Interest (POIs)** to candidate **addresses**, validates those relationships using **building footprint geometry**, and connects matched addresses to **street-level information**.

Overture datasets contain places, addresses, buildings, and transportation features as separate themes. Places are point representations of real-world entities, addresses can support conflation with places and buildings, and buildings provide footprint geometries. This project builds a baseline relationship table connecting those entities in an interpretable way.

The main implementation is in:

```text
V3_final_problem1_pipeline/
```

The pipeline is designed to be **region-agnostic**: it can run on any Overture extract with the required layers. It was tested on one regional extract, and the current results should be interpreted as a baseline test run rather than a global production benchmark.

## Problem Statement

POIs, addresses, buildings, and streets are represented as separate map entities. In the real world, these entities are connected:

- A POI has or uses an address.
- An address belongs to, is inside, or is near a building.
- A building relates to nearby street infrastructure.
- A POI may also be inside or near a building footprint.

Without explicit relationships between these entities, downstream tasks become harder, including search, navigation, map validation, ranking, address enrichment, and knowledge graph construction.

This project builds a baseline method for creating those relationships from raw Overture data.

## Overture Data Sources

This pipeline uses the following Overture themes:

- Overture places
- Overture addresses
- Overture buildings
- Optional Overture street or segment data

Useful Overture documentation:

- [Getting Overture data](https://docs.overturemaps.org/getting-data/)
- [Places guide](https://docs.overturemaps.org/guides/places/)
- [Addresses guide](https://docs.overturemaps.org/guides/addresses/)
- [Buildings guide](https://docs.overturemaps.org/guides/buildings/)
- [Overture schema docs](https://docs.overturemaps.org/schema/)

## What This Pipeline Does

The V3 pipeline:

- Loads raw Overture places, addresses, and buildings.
- Supports Parquet, GeoJSON, and CSV where possible.
- Standardizes varying source schemas into consistent internal columns.
- Matches POIs to address candidates.
- Uses address text similarity when POI address text exists.
- Uses spatial nearest-address fallback when POI address text is missing.
- Matches POIs and matched addresses to building footprints.
- Validates whether the POI and matched address are in the same building.
- Extracts street names from matched addresses.
- Optionally supports Overture street or segment files if available.
- Produces final confidence labels and human-readable review reasons.

## Final Output

The main output is:

```text
V3_final_problem1_pipeline/outputs/final_problem1_conflation.csv
```

Each row represents a relationship chain:

```text
POI → matched address → matched building → matched street
```

This table can be used as a starting point for:

- Knowledge graph edges
- Map quality validation
- Search and ranking features
- Manual review workflows
- Address and building relationship analysis

## Example Relationship Representation

Example row representation:

```text
POI: Example Coffee Shop
matched_address_text: 123 MAIN ST
matched_building_id: building_x
matched_street: MAIN ST
final_label: same_building_confirmed
```

Graph-style edges:

```text
POI --located_at--> Address
Address --inside_or_near--> Building
Address --on_street--> Street
POI --inside_or_near--> Building
```

## Pipeline Architecture

```text
V3_final_problem1_pipeline/
  src/
    config.py
    data_loader.py
    normalization.py
    address_matcher.py
    building_matcher.py
    building_validator.py
    street_connector.py
    labeler.py
    summary_writer.py
    run_pipeline.py
  data/
    raw/
    processed/
  outputs/
  requirements.txt
```

Source files:

- `config.py`: Stores paths, input options, output paths, thresholds, sampling settings, and optional bounding box settings.
- `data_loader.py`: Loads raw Overture files and standardizes places, addresses, buildings, and optional street segments.
- `normalization.py`: Normalizes text, street names, house numbers, and address strings.
- `address_matcher.py`: Matches POIs to candidate addresses using text similarity, street similarity, and distance scoring.
- `building_matcher.py`: Matches POI and address points to building footprints using spatial joins and nearest-building fallback.
- `building_validator.py`: Assigns building validation labels based on building consistency, conflicts, and shared-address clustering.
- `street_connector.py`: Extracts matched street names and optionally compares them with nearest street segments.
- `labeler.py`: Produces final labels, match methods, POI class labels, and review reasons.
- `summary_writer.py`: Writes summary counts and a manual evaluation template.
- `run_pipeline.py`: Orchestrates the full end-to-end pipeline.

## How To Run

Install dependencies:

```bash
pip install -r V3_final_problem1_pipeline/requirements.txt
```

Place raw data files in:

```text
V3_final_problem1_pipeline/data/raw/
```

Required inputs:

```text
places.parquet OR places.geojson OR places.csv
addresses.parquet OR addresses.geojson OR addresses.csv
buildings.parquet OR buildings.geojson OR buildings.csv
```

Optional input:

```text
streets/segments file
```

Run the pipeline:

```bash
python V3_final_problem1_pipeline/src/run_pipeline.py
```

## Expected Outputs

The pipeline writes three main outputs:

- `final_problem1_conflation.csv`: Final relationship table linking POIs, addresses, buildings, and streets.
- `problem1_summary.csv`: Summary counts by final label, match method, POI class, street connection status, building status, and address match status.
- `problem1_manual_eval_template.csv`: A 50-row sample for manual review with empty `manual_result` and `manual_notes` columns.

## Output Columns

Important output columns include:

- `poi_id`
- `poi_name`
- `poi_lat`
- `poi_lon`
- `poi_address_input`
- `matched_address_id`
- `matched_address_text`
- `matched_address_lat`
- `matched_address_lon`
- `matched_building_id`
- `matched_street`
- `nearest_street_segment_id`
- `nearest_street_name`
- `distance_m`
- `address_similarity`
- `street_similarity`
- `confidence`
- `address_match_status`
- `building_status`
- `real_building_relation`
- `building_validation_label`
- `poi_class`
- `match_method`
- `street_connection_status`
- `final_label`
- `review_reason`

## Label Meanings

- `same_building_confirmed`: The POI and matched address are supported by strong building evidence, usually the same building footprint.
- `building_validated_candidate`: The POI-address match is supported by building validation, but the evidence is not strong enough for the highest-confidence label.
- `spatial_address_candidate`: The POI does not have address text, but a nearby address candidate was found using spatial evidence.
- `missing_poi_address_no_candidate`: The POI has no address text and no nearby address candidate was found.
- `non_building_poi`: The POI appears to represent a non-building entity such as a park, trail, lake, viewpoint, or other outdoor/non-addressed feature.
- `text_and_distance_match`: The POI address text and distance both support the address match.
- `needs_review`: The evidence is weak, incomplete, or conflicting and should be manually reviewed.

## Current Tested Result

On one regional Overture extract, the standalone V3 pipeline loaded:

- 3,545 POIs
- 10,332 addresses
- 9,699 buildings

It produced:

- 1,516 `same_building_confirmed` rows
- 1,881 `needs_review` rows
- 148 `non_building_poi` rows

These numbers describe one test run on one regional extract. They are not intended as a universal benchmark for Overture data quality or conflation accuracy.

## Key Findings

- Many map entities are not explicitly linked across Overture themes.
- POI address text may be missing, incomplete, or inconsistent.
- Spatial matching alone can create false positives.
- Building geometry is useful for validating POI-address relationships.
- Non-building POIs should be separated from building-expected POIs before evaluating address/building matches.
- A confidence and review system is necessary for noisy real-world map data.

## Evaluation

The repository creates:

```text
V3_final_problem1_pipeline/outputs/problem1_manual_eval_template.csv
```

This file samples rows from the final output and adds fields for manual review:

```text
manual_result
manual_notes
```

Reviewers can label rows as:

```text
correct, incorrect, unsure, not_applicable
```

If an official ground truth dataset is provided later, this pipeline can be evaluated more formally using precision, recall, and error analysis by label type.

## Design Choices

- **Rule-based baseline instead of ML**: The current pipeline prioritizes interpretability and transparent review reasons.
- **Region-agnostic paths**: Raw inputs are loaded from `data/raw/` without hardcoded regional assumptions.
- **Confidence labels instead of only binary matches**: Real-world map data often contains partial or conflicting evidence.
- **Optional street segment support**: The pipeline runs without street segments but can use them if available.
- **Git-friendly data handling**: Raw data and generated outputs are ignored from git to keep the repository lightweight.

## Limitations

- Accuracy depends on the completeness and quality of the Overture extract.
- Some POIs do not include address text.
- Malls, campuses, shared addresses, and multi-tenant buildings are difficult to resolve with simple rules.
- Street connection is currently address-derived unless a street or segment file is provided.
- No official ground truth dataset is currently included.
- The project is a baseline research/engineering pipeline, not a global-scale production conflation system.

## Future Work

- Add official ground truth evaluation if available.
- Improve handling of multi-tenant buildings, malls, campuses, and shared commercial addresses.
- Use Overture transportation segments more deeply for street validation.
- Add map visualization for manual inspection.
- Export graph edges directly.
- Improve confidence scoring with learned models in a later version.
- Scale testing across more regions and compare regional error patterns.

## Git And Data Note

Raw Overture data and generated outputs are intentionally not committed. Users should download Overture extracts, place them in:

```text
V3_final_problem1_pipeline/data/raw/
```

and then run the standalone V3 pipeline locally.
