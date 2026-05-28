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

To build and evaluate against the Boulder/Louisville, Colorado proxy benchmark geography:

```bash
python Problem2_facade_pipeline/src/overture_proxy_data_builder.py --overwrite
$env:USE_PROXY_AREA_OVERTURE_DATA='1'; python Conflation_pipeline/src/run_pipeline.py
$env:USE_PROXY_AREA_OVERTURE_DATA='1'; python Problem2_facade_pipeline/src/run_pipeline.py
```

For proxy benchmarking only, `needs_review` rows can be allowed through facade candidate generation:

```bash
$env:USE_PROXY_AREA_OVERTURE_DATA='1'; $env:PROXY_EVAL_INCLUDE_NEEDS_REVIEW='1'; python Conflation_pipeline/src/run_pipeline.py
$env:USE_PROXY_AREA_OVERTURE_DATA='1'; $env:PROXY_EVAL_INCLUDE_NEEDS_REVIEW='1'; python Problem2_facade_pipeline/src/run_pipeline.py
```

## Input Files

| Input | Purpose |
| --- | --- |
| `Conflation_pipeline/outputs/final_problem1_conflation.csv` | Problem 1 relationship and label signals. |
| `Problem2_facade_pipeline/data/raw/buildings.geojson` | Building polygons used to extract facade edges. |
| `Problem2_facade_pipeline/data/raw/segments.geojson` | Street geometry used for street-aware re-ranking. |
| `Problem2_facade_pipeline/data/proxy_overture/proxy_area_buildings.parquet` | Optional proxy-area building extract when `USE_PROXY_AREA_OVERTURE_DATA=1`. |
| `Problem2_facade_pipeline/data/proxy_overture/proxy_area_places.parquet` | Optional proxy-area POI extract for rerunning Problem 1 on the proxy benchmark geography. |
| `Problem2_facade_pipeline/data/proxy_overture/proxy_area_addresses.parquet` | Optional proxy-area address extract for rerunning Problem 1 on the proxy benchmark geography. |
| `Problem2_facade_pipeline/data/proxy_overture/proxy_area_segments.parquet` | Optional proxy-area street segment extract for reranking on the proxy benchmark geography. |
| `benchmark_dataset/reference_datasets/extracted/boulder_louisville_sign_and_entrance_gt/boulder_snips_gt_adjusted.csv` | Boulder sign/front-door proxy storefront points. |
| `benchmark_dataset/reference_datasets/extracted/boulder_louisville_sign_and_entrance_gt/louisville_snips_gt_adjusted.csv` | Louisville, Colorado sign/front-door proxy storefront points. |

## Output Files

| Output | Description |
| --- | --- |
| `Problem2_facade_pipeline/outputs/problem2_facade_matches.csv` | One output row per Problem 1 row, including status and selected facade fields. |
| `Problem2_facade_pipeline/outputs/problem2_summary.csv` | Counts and average facade metrics from the run. |
| `Problem2_facade_pipeline/benchmark_dataset/problem2_proxy_candidates.csv` | Cleaned entrance/sign proxy candidate rows. |
| `Problem2_facade_pipeline/benchmark_dataset/problem2_proxy_matched_rows.csv` | Proxy rows matched to Problem 2 outputs. |
| `Problem2_facade_pipeline/benchmark_dataset/problem2_proxy_unmatched_rows.csv` | Proxy rows that could not be safely matched to Problem 2 outputs. |
| `Problem2_facade_pipeline/benchmark_dataset/problem2_proxy_matched_not_evaluated_rows.csv` | Matched proxy rows skipped because no selected facade was available. |
| `Problem2_facade_pipeline/outputs/problem2_proxy_evaluation.csv` | Row-level proxy facade agreement results. |
| `Problem2_facade_pipeline/outputs/problem2_proxy_summary.csv` | Aggregate proxy agreement metrics. |
| `Problem2_facade_pipeline/outputs/problem2_proxy_ranker_comparison.csv` | Nearest-edge baseline vs selected/reranked proxy agreement comparison. |
| `Problem2_facade_pipeline/benchmark_dataset/problem2_proxy_reranker_improved_rows.csv` | Rows where selected/reranked facade agrees with proxy but nearest-edge baseline does not. |
| `Problem2_facade_pipeline/benchmark_dataset/problem2_proxy_baseline_correct_reranker_wrong_rows.csv` | Rows where nearest-edge baseline agrees with proxy but selected/reranked facade does not. |
| `Problem2_facade_pipeline/benchmark_dataset/problem2_proxy_needs_review_evaluated_rows.csv` | Permissive proxy-mode evaluated rows that came from `needs_review` Problem 1 labels. |
| `Problem2_facade_pipeline/benchmark_dataset/problem2_proxy_confirmed_evaluated_rows.csv` | Evaluated rows that did not require permissive `needs_review` inclusion. |
| `Problem2_facade_pipeline/benchmark_dataset/problem2_proxy_reranker_correct_baseline_wrong.csv` | Alias output for rows where selected/reranked facade agrees with proxy but nearest-edge baseline does not. |
| `Problem2_facade_pipeline/benchmark_dataset/problem2_proxy_baseline_correct_reranker_wrong.csv` | Alias output for rows where nearest-edge baseline agrees with proxy but selected/reranked facade does not. |

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

Street geometry was available in the current proxy-area run, so `street_used_for_rerank` is `true` for the 753 facade candidate rows and `selected_method` is `street_aware_rerank`.

## Proxy-Area Overture Extraction

The first proxy evaluation exposed a geography mismatch: the Louisville proxy rows are for Louisville, Colorado, but the earlier local Overture extract included a different Louisville coordinate range around longitude `-85`. Those Louisville, Colorado proxy rows were intentionally left unmatched rather than force-matched.

The additive fix is `Problem2_facade_pipeline/src/overture_proxy_data_builder.py`. It reads the actual proxy benchmark coordinates, detects the latitude/longitude columns, computes buffered bounding boxes for Boulder and Louisville, Colorado, and downloads Overture data directly for those proxy areas.

Detected proxy bounding boxes from the source coordinates:

| Proxy Area | Proxy Rows | Bounding Box |
| --- | ---: | --- |
| Boulder | 349 | `-105.2845424,40.0160823,-105.2702857,40.0208041` |
| Louisville, Colorado | 254 | `-105.1336847,39.9750387,-105.127107,39.9814959` |

Proxy-area Overture extract created:

| Layer | Rows | Output |
| --- | ---: | --- |
| Places | 1,798 | `Problem2_facade_pipeline/data/proxy_overture/proxy_area_places.parquet` |
| Buildings | 1,078 | `Problem2_facade_pipeline/data/proxy_overture/proxy_area_buildings.parquet` |
| Addresses | 1,436 | `Problem2_facade_pipeline/data/proxy_overture/proxy_area_addresses.parquet` |
| Street segments | 604 | `Problem2_facade_pipeline/data/proxy_overture/proxy_area_segments.parquet` |

These proxy-area data files are generated local data and should not be committed. They are ignored by `.gitignore`.

## Proxy Evaluation With Entrance And Sign Points

Because verified facade-edge annotations were not available, entrance and sign coordinates from the Louisville/Boulder datasets are used as proxy storefront indicators. The facade edge closest to the entrance/sign point is treated as a proxy facade label for evaluation. These metrics are proxy agreement metrics, not final ground-truth accuracy.

The proxy workflow runs after normal Problem 2 matching:

1. Load Boulder/Louisville sign and entrance CSVs.
2. Keep usable, non-excluded proxy points with coordinates.
3. Match proxy rows to Problem 2 output using Overture POI id first, then conservative name plus coordinate fallback.
4. Split the matched building polygon into facade edges.
5. Find the edge closest to the entrance/sign point.
6. Compare that proxy edge with the selected Problem 2 facade edge.

This gives a stronger evaluation signal than ambiguity reduction alone because it uses field-observed storefront-related points. It is still not a true accuracy score because the proxy point nearest edge may differ from a human-verified storefront facade label.

## Permissive Proxy Evaluation Mode

By default, Problem 2 only evaluates rows that pass the normal upstream Problem 1 filtering. In the proxy-area run, many rows matched to Overture data but remained unevaluated because Problem 1 labeled them `needs_review`, so they did not receive selected facades.

For proxy benchmarking only, `PROXY_EVAL_INCLUDE_NEEDS_REVIEW=1` allows rows with `final_label == "needs_review"` to enter facade candidate generation. This increases evaluation coverage, but these rows should not be treated as production-confirmed matches. The output marks this clearly with `problem2_eval_mode`, `source_final_label`, and `proxy_eval_included_needs_review`.

Metrics remain proxy metrics using entrance/sign points as proxy ground truth. This mode is useful for testing facade-selection behavior on more matched rows, not for claiming final human-labeled accuracy.

## Reranker Tuning and Non-Building POI Filtering

Facade matching only applies to POIs attached to buildings. Proxy evaluation therefore excludes rows that Problem 1 or the POI type signals as non-building, including `non_building_poi`, `missing_poi_address_no_candidate`, parks, trails, outdoor areas, open spaces, transit stops, and similar features. These excluded rows are written to `Problem2_facade_pipeline/benchmark_dataset/problem2_proxy_non_building_pois_excluded.csv`.

The previous street-aware reranker underperformed the nearest-edge baseline on the permissive proxy benchmark, suggesting that street/orientation cues were sometimes overpowering POI-to-edge distance. The additive tuning script `Problem2_facade_pipeline/src/tune_facade_reranker.py` grid-searches distance-dominant reranker settings while keeping the original reranker available. Distance remains the strongest signal, and street orientation is treated as a secondary cue for close or ambiguous candidates.

Tuning outputs:

| Output | Description |
| --- | --- |
| `Problem2_facade_pipeline/outputs/problem2_reranker_tuning_results.csv` | Grid-search results for reranker weight/guardrail combinations. |
| `Problem2_facade_pipeline/outputs/problem2_best_reranker_config.json` | Best proxy-benchmark configuration from the grid search. |
| `Problem2_facade_pipeline/benchmark_dataset/problem2_proxy_evaluated_building_based_rows.csv` | Building-based rows used for proxy accuracy. |

Current building-only proxy evaluation before tuning:

| Metric | Value |
| --- | ---: |
| Usable proxy rows | 603 |
| Non-building POIs excluded | 30 |
| Matched building-based rows | 454 |
| Evaluated building-based rows | 450 |
| Selected/reranked proxy accuracy | 46.8889% |
| Baseline nearest-edge proxy accuracy | 53.1111% |
| Confirmed/high-confidence proxy accuracy | 35.8025% |
| `needs_review` proxy accuracy | 49.3225% |

Best tuning result from `problem2_reranker_tuning_results.csv`:

| Metric | Value |
| --- | ---: |
| Best selected/reranked proxy accuracy | 54.6667% |
| Baseline nearest-edge proxy accuracy | 53.1111% |
| Improvement over baseline | 1.5556 percentage points |
| Improved / worsened / unchanged | 9 / 2 / 439 |
| Boulder evaluated rows | 254 |
| Louisville evaluated rows | 196 |

Best parameters from `problem2_best_reranker_config.json`:

| Parameter | Value |
| --- | ---: |
| `distance_weight` | 0.75 |
| `street_alignment_weight` | 0.06 |
| `street_facing_weight` | 0.0 |
| `edge_length_weight` | 0.0 |
| `corner_penalty_weight` | 0.2 |
| `shared_facade_penalty_weight` | 0.0 |
| `lockin_distance_m` | 5.0 |
| `max_distance_ratio_for_override` | 2.0 |
| `strong_override_margin` | 0.1 |

The tuned reranker is not enabled by default. To activate it for an experimental run, set `TUNE_FACADE_RERANKER=1`. Reported metrics are still proxy agreement metrics using entrance/sign points, not final human-verified ground truth accuracy.

## Accuracy Breakdown and Honest Higher-Quality Subsets

Overall building-only proxy accuracy is the broadest current metric. High-confidence and clean subsets are reported separately because they represent rows where facade matching is more appropriate or upstream building linkage is stronger. Rows are not removed silently: every subset has row counts and exclusion reasons.

`needs_review` rows remain useful for stress-testing the facade selector, but they are not production-confirmed POI-building matches. Shared mall/plaza buildings are also difficult because many POIs can share one footprint, and real storefront boundaries may not exist as separate Overture building edges.

Current tuned full-data benchmark from `problem2_reranker_tuning_results.csv`:

| Group | Rows | Tuned Accuracy | Baseline Accuracy | Difference |
| --- | ---: | ---: | ---: | ---: |
| all building-based | 450 | 54.6667% | 53.1111% | +1.5556 points |
| high-confidence building-linked | 81 | 32.0988% | 33.3333% | -1.2346 points |
| `needs_review` only | 369 | 59.6206% | 57.4526% | +2.1680 points |
| clean building proxy subset | 145 | 52.4138% | 48.2759% | +4.1379 points |
| clean high-confidence subset | 24 | 41.6667% | 41.6667% | 0.0000 points |
| shared-building rows | 449 | 54.7884% | 53.2294% | +1.5590 points |

No honest reported subset exceeded 60% in this run. The closest was `needs_review` only at 59.6206%, but that subset should not be presented as production-confirmed accuracy.

Deterministic 70/30 train/test tuning from `problem2_reranker_tuning_train_test_results.csv`:

| Metric | Value |
| --- | ---: |
| Train rows | 315 |
| Test rows | 135 |
| Best train accuracy | 56.1905% |
| Held-out test accuracy | 51.1111% |
| Held-out test baseline | 49.6296% |
| Held-out test improvement | +1.4815 points |
| High-confidence test accuracy | 28.5714% |
| `needs_review` test accuracy | 55.2632% |

If a future subset exceeds 60%, it should be stated only for that subset, not as overall Problem 2 accuracy.

Untuned permissive proxy-mode summary recorded before enabling `TUNE_FACADE_RERANKER`, with `USE_PROXY_AREA_OVERTURE_DATA=1`, `PROXY_EVAL_INCLUDE_NEEDS_REVIEW=1`, and non-building proxy filtering enabled:

| Metric | Value |
| --- | ---: |
| Total proxy rows loaded | 612 |
| Usable proxy rows | 603 |
| Non-building POIs excluded | 30 |
| Matched to Problem 2 outputs | 484 |
| Matched building-based rows | 454 |
| Unmatched rows | 119 |
| Matched but not evaluated rows | 4 |
| Evaluated rows | 450 |
| Evaluated confirmed/high-confidence rows | 81 |
| Evaluated `needs_review` rows | 369 |
| Selected/reranked proxy accuracy overall | 46.8889% |
| Proxy accuracy excluding `needs_review` | 35.8025% |
| Proxy accuracy for `needs_review` only | 49.3225% |
| Baseline nearest-edge proxy accuracy overall | 53.1111% |
| Proxy coverage / recall | 78.5340% |
| Matched coverage | 79.2321% |
| Evaluable among matched | 99.1189% |

Nearest-edge baseline vs selected/reranked comparison in permissive proxy mode:

| Metric | Count | Rate |
| --- | ---: | ---: |
| Baseline nearest-edge proxy agreement | 239 | 53.1111% |
| Selected/reranked facade proxy agreement | 211 | 46.8889% |
| Improvement count | 34 | 7.5556% |
| Worsened count | 62 | 13.7778% |
| Unchanged count | 354 | 78.6667% |

In permissive proxy mode with non-building filtering, evaluated rows include 254 Boulder rows and 196 Louisville, Colorado rows. Of those, 369 evaluated rows were included only because `PROXY_EVAL_INCLUDE_NEEDS_REVIEW=1`.

## Actual Result Summary

From `Problem2_facade_pipeline/outputs/problem2_summary.csv` after running with `USE_PROXY_AREA_OVERTURE_DATA=1`:

| Metric | Value |
| --- | ---: |
| Total Problem 1 rows consumed | 1,798 |
| `not_applicable` count | 45 |
| `no_building_match` count | 7 |
| `needs_review` count | 993 |
| `facade_candidate` count | 753 |
| `nearest_edge_baseline` count | 0 |
| `street_aware_rerank` count | 753 |
| Average nearest edge distance | 5.030 m |
| Average selected facade distance | 11.128 m |
| Average selected facade score | 0.43881 |

Before proxy-area extraction, the proxy evaluation matched 171 rows and evaluated 106 Boulder-only rows. After proxy-area extraction, the benchmark matched 484 rows across Boulder and Louisville, Colorado, but only 81 rows were evaluable because many matched rows were labeled `needs_review` by Problem 1 and therefore did not receive selected facades.

Strict proxy evaluation summary from `Problem2_facade_pipeline/outputs/problem2_proxy_summary.csv` before enabling permissive proxy mode:

| Metric | Value |
| --- | ---: |
| Total proxy rows loaded | 612 |
| Usable proxy rows | 603 |
| Matched to Problem 2 outputs | 484 |
| Unmatched rows | 119 |
| Matched but not evaluated rows | 403 |
| Evaluated rows | 81 |
| Exact proxy facade agreement count | 29 |
| Exact proxy facade agreement rate | 35.8025% |
| Proxy coverage / recall | 13.4328% |
| Matched coverage | 80.2653% |
| Evaluable among matched | 16.7355% |
| Average selected facade distance to proxy point | 11.222 m |

Agreement by proxy point type:

| Proxy Point Type | Evaluated Rows | Agreement Rate |
| --- | ---: | ---: |
| `sign` | 48 | 39.5833% |
| `front_door` | 32 | 31.25% |
| `entrance` | 1 | 0.0% |

Nearest-edge baseline vs selected/reranked comparison from `Problem2_facade_pipeline/outputs/problem2_proxy_ranker_comparison.csv`:

| Metric | Count | Rate |
| --- | ---: | ---: |
| Baseline nearest-edge proxy agreement | 27 | 33.3333% |
| Selected/reranked facade proxy agreement | 29 | 35.8025% |
| Improvement count | 12 | 14.8148% |
| Worsened count | 10 | 12.3457% |
| Unchanged count | 59 | 72.8395% |

In this proxy-area run, both Boulder and Louisville, Colorado rows are matched. Evaluated rows include 69 Boulder rows and 12 Louisville, Colorado rows. The low Louisville evaluated count is mostly a Problem 1 filtering issue: many Louisville proxy rows matched to Problem 2 output but were not evaluated because their Problem 1 rows were labeled `needs_review`.

## Output Columns

`problem2_facade_matches.csv` includes:

```text
poi_id
poi_name
problem1_final_label
source_final_label
problem2_eval_mode
proxy_eval_included_needs_review
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
- Entrance/sign proxy agreement is not the same as true facade accuracy.
- Proxy-area extraction fixes the Louisville, Colorado geography mismatch, but many matched rows are still filtered out as `needs_review` before facade evaluation.
- Street geometry helps, but it does not prove storefront correctness.
- Multi-tenant and corner buildings remain difficult.

## Future Work

- Add verified facade labels from sign and entrance reference data.
- Use entrance points directly when available.
- Improve street-name matching between Problem 1 and segment geometry.
- Evaluate nearest-edge baseline vs street-aware re-ranking against manual labels.
- Add visual QA packets for selected facades.
- Build a true golden dataset with manually verified building edge labels.
