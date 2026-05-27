# Problem 2: POI To Building Facade Matching

This pipeline extends Problem 1 by selecting the most likely building facade edge for high-confidence, building-based POIs.

## What Problem 2 Solves

Problem 1 links a POI to a building. Problem 2 asks a more detailed question:

> Which edge or side of that building is most likely related to the POI storefront?

The output is a facade match table for rows that pass the Problem 1 filter.

## Dependency On Problem 1

Problem 2 consumes:

```text
Conflation_pipeline/outputs/final_problem1_conflation.csv
```

It uses actual Problem 1 columns including:

- `poi_id`
- `poi_name`
- `poi_lat`
- `poi_lon`
- `matched_building_id`
- `matched_address_text`
- `matched_street`
- `street_segment_distance_m`
- `building_validation_label`
- `poi_class`
- `final_label`

Problem 2 does not facade-match every raw POI. It first classifies each Problem 1 row.

## Filtering Logic

| Problem 2 Status | Rule |
| --- | --- |
| `not_applicable` | `final_label` is `non_building_poi` or `poi_class` is `non_building_poi`. |
| `no_building_match` | No usable matched building or building geometry is available. |
| `needs_review` | Problem 1 label or building validation is weak, conflicting, or uncertain. |
| `facade_candidate` | Problem 1 has an acceptable building-based match. |

For this run, only `same_building_confirmed` and `building_validated_candidate` rows continued to facade matching.

## How To Run

Run Problem 1 first:

```bash
python Conflation_pipeline/src/run_pipeline.py
```

Then run Problem 2:

```bash
python Problem2_facade_pipeline/src/run_pipeline.py
```

## Input Files

| Input | Purpose |
| --- | --- |
| `Conflation_pipeline/outputs/final_problem1_conflation.csv` | Problem 1 relationship and label signals. |
| `Problem2_facade_pipeline/data/raw/buildings.geojson` | Building polygons used to extract facade edges. |
| `Problem2_facade_pipeline/data/raw/segments.geojson` | Street geometry used for street-aware re-ranking. |

## Output Files

| Output | Description |
| --- | --- |
| `Problem2_facade_pipeline/outputs/problem2_facade_matches.csv` | One output row per Problem 1 row, including status and selected facade fields. |
| `Problem2_facade_pipeline/outputs/problem2_summary.csv` | Counts and average facade metrics from the run. |

## Facade Candidate Generation

For each `facade_candidate` row:

1. Load the matched building polygon.
2. Split the polygon exterior into individual edge segments.
3. Compute edge id, edge length, edge bearing, POI-to-edge distance, projected POI position, and corner proximity.
4. Select the nearest edge as the baseline.
5. Apply street-aware re-ranking when street geometry is available.

## Nearest-Edge Baseline

The baseline chooses the building edge with the shortest distance to the POI point.

Output baseline fields include:

- `nearest_edge_id`
- `nearest_edge_distance_m`
- `nearest_edge_bearing`
- `nearest_edge_length_m`

## Street-Aware Re-Ranking

Nearest edge alone can choose a back wall, side wall, alley-facing wall, or corner fragment.

The improvement uses these signals:

- POI-to-edge distance
- edge distance to nearby street geometry
- edge alignment with nearby street geometry
- edge length
- corner penalty
- shared/crowded facade penalty

Street geometry was available in this run, so `street_used_for_rerank` is `true` for the 2,815 facade candidate rows and `selected_method` is `street_aware_rerank`.

## Actual Result Summary

From `Problem2_facade_pipeline/outputs/problem2_summary.csv`:

| Metric | Value |
| --- | ---: |
| Total Problem 1 rows consumed | 4,039 |
| `not_applicable` count | 82 |
| `no_building_match` count | 2 |
| `needs_review` count | 1,140 |
| `facade_candidate` count | 2,815 |
| `nearest_edge_baseline` count | 0 |
| `street_aware_rerank` count | 2,815 |
| Average nearest edge distance | 7.141 m |
| Average selected facade distance | 17.244 m |
| Average selected facade score | 0.439624 |

## Output Columns

`problem2_facade_matches.csv` includes:

```text
poi_id
poi_name
problem1_final_label
problem2_status
building_id
nearest_edge_id
nearest_edge_distance_m
nearest_edge_bearing
nearest_edge_length_m
selected_facade_edge_id
selected_facade_distance_m
selected_facade_score
selected_method
selected_facade_reason
street_used_for_rerank
street_alignment_score
street_facing_score
facade_edge_length_m
facade_edge_bearing
corner_penalty
shared_facade_penalty
```

## Limitations

- This is still a rule-based baseline and prototype.
- The selected street-aware edge can be farther from the POI than the nearest edge because public-facing street context is part of the score.
- No human-verified facade labels are included yet.
- Street geometry helps, but it does not prove storefront correctness.
- Multi-tenant and corner buildings remain difficult.

## Future Work

- Add verified facade labels from sign and entrance reference data.
- Use entrance points directly when available.
- Improve street-name matching between Problem 1 and segment geometry.
- Evaluate nearest-edge baseline vs street-aware re-ranking against manual labels.
- Add visual QA packets for selected facades.
