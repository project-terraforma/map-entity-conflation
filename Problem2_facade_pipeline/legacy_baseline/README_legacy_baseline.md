# Problem 2 - POI to Building Facade Matching

This folder contains a baseline pipeline for Overture Problem 2:

> Estimate which building footprint side or edge is most likely related to a POI storefront.

The pipeline is intentionally schema-adaptive. It prints the files, columns, and sample rows it actually finds before choosing what evidence to use. It does not assume that any Overture export contains a specific optional field.

## Local Data Inspection

At implementation time in this repository, no local `.parquet`, `.geojson`, or `.csv` Overture data files were present. The code therefore supports inspection and graceful failure when required layers are missing.

The runner looks for data in this order:

1. `Problem2_facade_pipeline/data/raw/`
2. `data/raw/`
3. `Conflation_pipeline/data/raw/`

It also looks for Problem 1 output in:

```text
Conflation_pipeline/outputs/final_problem1_conflation.csv
```

If those files are absent, the pipeline writes `outputs/problem2_summary.csv` with missing-layer notes instead of pretending a match was produced.

## Inputs Used When Available

Required for facade matching:

- `places`: POI point geometry, or latitude/longitude columns.
- `buildings`: polygon or multipolygon building footprints.

Optional evidence:

- Problem 1 output with `poi_id` and a matched building id.
- Place-level building id fields if present.
- Spatial nearest-building fallback within a conservative threshold.
- `entrances` or `connectors` point layer.
- `streets` or `segments` line layer.
- `addresses` point layer is inspected and standardized for future use, but the baseline facade choice does not depend on it unless later extended.

## Method

For each POI:

1. Attach the best available building id from Problem 1 output, a place/building column, or nearest building fallback.
2. Reproject POIs, buildings, entrances, and streets to an estimated local metric CRS.
3. Split the matched building exterior ring into individual facade edge segments.
4. Compute POI-to-edge distance for every candidate facade.
5. Choose the nearest facade edge by default.
6. If entrance points exist, choose the facade closest to the nearest entrance to the POI.
7. Add optional street support by measuring selected-facade distance to the street/segment layer.
8. Score the match with an interpretable confidence label and evidence notes.

## Confidence Labels

The baseline can emit:

- `high_confidence_entrance_supported`
- `high_confidence_nearest_facade`
- `medium_confidence_inside_building_nearest_edge`
- `medium_confidence_street_supported`
- `medium_confidence_nearest_facade`
- `low_confidence_far_from_building`
- `needs_review_no_building_match`
- `needs_review_invalid_geometry`
- `needs_review_multiple_close_facades`

## Outputs

The pipeline writes:

```text
outputs/problem2_facade_matches.csv
outputs/problem2_facade_matches.geojson
outputs/problem2_summary.csv
outputs/manual_review_facade_sample.csv
outputs/problem2_debug_sample.geojson
```

Match rows include POI id/name/category when available, POI coordinates, matched building id, facade id/index, facade WKT, facade midpoint coordinates, distance to facade in meters, method used, confidence score/label, evidence fields used, and review notes.

The debug GeoJSON includes a small sample of POI points, building footprints, selected facade edges, and all candidate facade edges for visual inspection.

## How To Run

Install dependencies:

```bash
pip install -r Problem2_facade_pipeline/legacy_baseline/requirements_legacy.txt
```

Add raw Overture extracts to one of the input folders listed above, then run:

```bash
python Problem2_facade_pipeline/legacy_baseline/run_pipeline.py
```

The script prints detected files, columns, sample rows, usable geometry counts, and match totals.

## Limitations

- This is a rule-based baseline, not a trained storefront detector.
- Building footprints do not always identify true storefront frontage.
- POIs inside large malls or multi-tenant buildings may have several plausible facades.
- Entrance and street support are only used when those layers exist locally.
- Address data is inspected but not yet used as a strong facade signal.
- Accuracy depends heavily on the quality and completeness of the local Overture extract.

## Future Improvements

- Use address points to infer frontage for addressable storefronts.
- Associate entrances to buildings and POIs more explicitly.
- Add road-facing orientation checks instead of simple road distance.
- Handle indoor/mall POIs and multi-tenant buildings as a separate class.
- Add batch map tiles or HTML visualization for manual QA.
- Evaluate against hand-labeled facade matches.
