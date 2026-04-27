# Project Zephr Baseline

`project_zephr_baseline` is a transparent baseline conflation pipeline for linking Overture `places`, `buildings`, `addresses`, and `transportation` road features inside a user-provided study polygon. The goal is to solve Zephr Problem 1: build a clean, explainable spatial relationship layer that ties POIs to building footprints, likely addresses, enclosing roads, and an estimated entrance-facing point that can later support image and embedding enrichment.

This baseline treats conflation as a geospatial entity resolution problem. It separates:

1. candidate generation
2. feature extraction
3. rule-based scoring
4. relationship resolution
5. evaluation and visual inspection

## Why Candidate Generation Matters

Directly matching every place to every building, address, or road is both expensive and noisy. Candidate generation narrows the search space to plausible nearby pairs based on containment and proximity, then feature engineering and scoring decide which of those plausible pairs should be linked.

## Relationship Tables

The pipeline writes four main outputs:

- `address_to_building`: primary building assignment for each address point
- `place_to_building`: primary building assignment for each Overture place / POI
- `building_to_street`: primary road frontage assignment for each building, with optional secondary roads
- `place_to_address`: primary address assignment for each place

Each output includes `score`, `confidence_tier`, and `match_reason`. GeoParquet exports are produced where geometry is useful for inspection.

## Scoring Approach

Scores are weighted heuristics in the range `[0, 1]`, not machine learning. The baseline emphasizes:

- containment bonuses when a point falls inside a building footprint
- inverse-distance style proximity scores
- street-name and house-number agreement where available
- relational support from already resolved upstream links

Weights and thresholds live in [config/settings.yaml](./config/settings.yaml).

## Repository Layout

```text
project_zephr_baseline/
├── README.md
├── requirements.txt
├── pyproject.toml
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── outputs/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── candidate_generation.py
│   ├── config.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── io_overture.py
│   ├── main.py
│   ├── preprocessing.py
│   ├── resolution.py
│   ├── scoring.py
│   ├── utils.py
│   └── visualization.py
└── tests/
    ├── test_features.py
    ├── test_resolution.py
    └── test_scoring.py
```

## Input Data

Required:

- `study_area.geojson` containing one polygon or multipolygon

Optional:

- one or more QA/reference CSVs with columns like `POI_name`, `overture_id`, `Type`, `Latitude`, `Longitude`, `Exclude`

Overture inputs can be supplied as:

- local GeoParquet, GeoPackage, FlatGeobuf, GeoJSON, or shapefiles
- public Overture parquet paths on S3 such as `s3://overturemaps-us-west-2/release/...`

For remote Overture parquet, the pipeline uses DuckDB over S3 and applies a study-area bounding-box filter before loading rows into GeoPandas.

Expected Overture themes:

- `places`: point POIs
- `buildings`: polygon footprints
- `addresses`: point addresses
- `roads`: line features from the transportation theme

## Running The Pipeline

### 1. Create an environment

```bash
cd project_zephr_baseline
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Point the config at your data

Edit [config/settings.yaml](./config/settings.yaml) and set:

- `study_area_path`
- `inputs.places_path`
- `inputs.buildings_path`
- `inputs.addresses_path`
- `inputs.roads_path`

Example for Boulder with direct Overture S3 paths:

```yaml
study_area_path: ../boulder_louisville_sign_and_entrance_gt/boulder_medium.geojson
reference_csv_paths:
  - ../boulder_louisville_sign_and_entrance_gt/boulder_snips_gt_adjusted.csv

inputs:
  places_path: s3://overturemaps-us-west-2/release/2026-03-18.0/theme=places/type=place/*.parquet
  buildings_path: s3://overturemaps-us-west-2/release/2026-03-18.0/theme=buildings/type=building/*.parquet
  addresses_path: s3://overturemaps-us-west-2/release/2026-03-18.0/theme=addresses/type=address/*.parquet
  roads_path: s3://overturemaps-us-west-2/release/2026-03-18.0/theme=transportation/type=segment/*.parquet
  write_clipped_extracts: true
```

You can also use local file paths instead.

Example for Louisville:

```yaml
study_area_path: ../boulder_louisville_sign_and_entrance_gt/louisville_medium.geojson
reference_csv_paths:
  - ../boulder_louisville_sign_and_entrance_gt/louisville_snips_gt_adjusted.csv
```

### 3. Run

```bash
python -m src.main --config config/settings.yaml
```

Or after editable install:

```bash
pip install -e .
zephr-baseline --config config/settings.yaml
```

## Main Outputs

Outputs are written under `data/outputs/`:

- `address_to_building.csv`
- `address_to_building.parquet`
- `place_to_building.csv`
- `place_to_building.parquet`
- `building_to_street.csv`
- `building_to_street.parquet`
- `place_to_address.csv`
- `place_to_address.parquet`
- `metrics_summary.json`
- `error_samples.csv`
- `figures/*.png`

Clipped per-layer extracts are optionally saved to `data/interim/`.

## Estimated Entrance Point

If a building is resolved to a primary road, the pipeline computes a simple estimated entrance point by projecting the building centroid to the nearest point on the selected road segment and choosing the closest facade point on the building exterior. This is intentionally rough, but it is a useful baseline aligned with Zephr’s embedding-tile idea of storing entrance coordinates beside building and road IDs.

## Current Limitations

- Remote Overture access depends on DuckDB's `httpfs` support and public S3 availability.
- Overture schemas vary across releases, so the code uses defensive column normalization and may still need minor adjustments for a new snapshot.
- Place-name support for `place_to_address` is deliberately weak and explainable.
- Corner lots and mixed-use buildings remain ambiguous in the rule baseline.
- Frontage and entrance inference are geometric proxies, not true facade or doorway detection.

## Next Steps

- add stronger text normalization and category-aware rules
- enrich road-facing facade estimation using parcel context and doorway imagery
- compare against human-labeled QA sets for precision and recall
- add embedding-based reranking after this transparent baseline is trusted

## Assumptions

- All distance calculations are done in a projected CRS, defaulting to `EPSG:26913` for Colorado.
- Missing Overture attributes are expected; the pipeline handles nulls conservatively.
- One primary building is resolved per address and place unless the score is too weak.
- One primary street is resolved per building, with optional secondary roads retained for debug analysis.
