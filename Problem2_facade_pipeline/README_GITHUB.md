# Problem 2: Facade / Storefront Matching

Problem 2 develops a storefront/facade matching baseline for POIs using Overture building footprints and entrance/sign proxy points.

Given a POI and its associated building footprint, the goal is to identify the most likely building facade or storefront side connected to that POI. This supports storefront and signage localization, entrance placement, accessibility routing, delivery navigation, and map-quality review.

## What This Pipeline Does

The pipeline builds and evaluates:

- facade candidate generation from Overture building footprints
- nearest-edge baseline
- street-aware reranker
- tuned reranker with distance guardrails
- proxy benchmark using entrance/sign/front-door points
- proxy-area Overture extraction for Boulder and Louisville, Colorado
- strict and permissive proxy evaluation modes
- non-building POI filtering
- label-group, clean-subset, shared-building, and city-level analyses
- local storefront heuristic and corner-ambiguity diagnostics
- train/test tuning analysis
- failure-analysis CSV outputs

Optional heuristic modes are off by default. Default production behavior is preserved.

## Data Sources

The current benchmark uses:

- Overture places / POIs
- Overture buildings
- Overture addresses, used by Problem 1 when available
- Overture street segments, used by the street-aware reranker
- Louisville/Boulder entrance and sign proxy benchmark rows

Proxy-area Overture extraction fixed an earlier geography mismatch: the Louisville proxy rows refer to Louisville, Colorado, while an earlier local extraction used a different Louisville coordinate range.

Proxy-area bounding boxes:

| Area | Bounding Box |
| --- | --- |
| Boulder | `-105.2845424,40.0160823,-105.2702857,40.0208041` |
| Louisville, Colorado | `-105.1336847,39.9750387,-105.127107,39.9814959` |

Extracted Overture rows:

| Layer | Rows |
| --- | ---: |
| Places | 1,798 |
| Buildings | 1,078 |
| Addresses | 1,436 |
| Street segments | 604 |

## Proxy Benchmark Method

Verified facade-edge annotations were not available. Entrance/sign/front-door points are used as proxy storefront indicators.

For each evaluated POI:

1. Load the matched building polygon.
2. Split the building exterior into facade edge candidates.
3. Find the facade edge closest to the entrance/sign point.
4. Treat that edge as the proxy facade label.
5. Compare the pipeline-selected facade against that proxy facade.

These are proxy agreement metrics, not human-labeled ground-truth facade accuracy.

## How To Run

Proxy-area extraction:

```powershell
python Problem2_facade_pipeline/src/overture_proxy_data_builder.py --overwrite
```

Strict/permissive building-only evaluation:

```powershell
$env:USE_PROXY_AREA_OVERTURE_DATA='1'
$env:PROXY_EVAL_INCLUDE_NEEDS_REVIEW='1'
$env:EXCLUDE_NON_BUILDING_POIS_FROM_PROXY_EVAL='1'
python Conflation_pipeline/src/run_pipeline.py
python Problem2_facade_pipeline/src/run_pipeline.py
```

Tuning:

```powershell
python Problem2_facade_pipeline/src/tune_facade_reranker.py
```

Optional tuned pipeline:

```powershell
$env:TUNE_FACADE_RERANKER='1'
python Problem2_facade_pipeline/src/run_pipeline.py
```

Optional experimental local storefront heuristic:

```powershell
$env:ENABLE_LOCAL_STOREFRONT_HEURISTICS='1'
python Problem2_facade_pipeline/src/run_pipeline.py
```

Optional experimental shared-building logic:

```powershell
$env:ENABLE_SHARED_BUILDING_FACADE_LOGIC='1'
python Problem2_facade_pipeline/src/run_pipeline.py
```

## Main Results

Final strict building-only proxy evaluation:

| Metric | Value |
| --- | ---: |
| Building-only evaluated rows | 450 |
| Boulder evaluated rows | 254 |
| Louisville, Colorado evaluated rows | 196 |
| Non-building POIs excluded | 30 |
| Baseline nearest-edge proxy accuracy | 239 / 450 = 53.1111% |
| Final tuned strict proxy accuracy | 246 / 450 = 54.6667% |
| Improvement over baseline | +1.5556 percentage points |
| Held-out test accuracy | 59.2593% |
| Held-out test baseline | 49.6296% |
| Held-out test improvement | +9.6297 percentage points |
| Mall/plaza subset accuracy | 57.3333% |

Corner ambiguity diagnostic:

| Metric | Value |
| --- | ---: |
| Corner-ambiguous rows | 276 / 450 |
| Strict proxy accuracy | 54.6667% |
| Corner-tolerant near-correct diagnostic | 71.5556% |

The corner-tolerant metric is diagnostic only. It is not the official strict proxy accuracy.

## KR Status

Objective 2: met

The project built a complete facade/storefront matching baseline, proxy benchmark, tuning workflow, city-level evaluation, and documented failure analysis.

| KR | Target | Result | Status |
| --- | --- | --- | --- |
| KR1 | >=60% accuracy on labeled dataset of >=50 POIs | 54.6667% strict proxy accuracy on 450 building-based rows | Partially met / near target |
| KR2 | <=10 percentage point variance across Boulder and Louisville | 1.0365 percentage point variance | Met |
| KR3 | 60-70% correctness and documented failure patterns | Strict proxy accuracy below 60%, but failure patterns strongly documented | Partially met |

City-level KR2 metrics:

| City | Rows | Strict Correct | Strict Proxy Accuracy |
| --- | ---: | ---: | ---: |
| Boulder | 254 | 140 | 55.1181% |
| Louisville, Colorado | 196 | 106 | 54.0816% |

```text
variance_pp = abs(55.1181 - 54.0816) = 1.0365 percentage points
KR2_status = met
```

## Failure Analysis

Observed failure patterns:

- shared buildings, malls, plazas, and strip malls are hard because many POIs share one building footprint
- Overture building edges do not represent true storefront boundaries
- long facades may contain multiple stores
- corner buildings can have multiple plausible facades
- entrance/sign proxy points may be noisy
- `needs_review` rows may contain weak upstream POI-building matches
- street/orientation signals can hurt when they overpower distance

Important interpretation:

- nearest-edge distance is a strong baseline
- early street-aware reranking over-weighted street/orientation and underperformed
- tuning improved over baseline by keeping distance dominant and using street alignment as a small secondary signal
- local storefront heuristics did not improve the full strict metric beyond 54.6667%, but train/test analysis showed promise
- shared-building logic should remain experimental and off by default

## Limitations

- Metrics are proxy agreement using entrance/sign points, not human-labeled facade truth.
- The proxy point nearest facade may differ from a manually verified storefront facade.
- Shared commercial buildings need stronger storefront-level evidence.
- Overture building footprints are often too coarse for individual storefront boundaries.
- Permissive evaluation includes `needs_review` rows for benchmarking, but those rows are not production-confirmed matches.

## Future Work

- Build a small human-labeled golden facade dataset.
- Add storefront-level segmentation for long shared facades.
- Use unit addresses, entrances, signs, and visual QA packets for shared buildings.
- Improve mall/plaza and indoor/outdoor walkway handling.
- Tune separate models for confirmed rows versus `needs_review` rows.
- Keep reporting strict accuracy separately from corner-tolerant diagnostics.
