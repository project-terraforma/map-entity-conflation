# Overture Map Entity Conflation and Storefront Evaluation

This repository is a research-oriented geospatial pipeline suite for linking
Overture map entities and evaluating storefront/facade localization from
building footprints, POIs, addresses, streets, and field-observed sign/entrance
evidence.

The repository contains three related Project Terraforma / Overture tasks:

1. **Problem 1: POI, address, building, and street conflation**
   - Links Overture places to likely addresses, buildings, and street context.
   - Produces an interpretable relationship table for downstream review.
2. **Problem 2: facade / storefront matching**
   - Uses Problem 1 building matches to select the most likely building edge for
     a storefront-facing POI.
   - Evaluates against a field-observed sign/entrance proxy benchmark.
3. **Problem 3: imagery-seeded POI verification and enrichment**
   - Turns sign or entrance observations from imagery into conservative,
     reviewable candidate POI additions.
   - Separates local Overture existence checks from licensed external evidence.

The code emphasizes transparent heuristics, auditable intermediate outputs, and
explicit evaluation boundaries. In particular, the facade results reported here
are **proxy agreement metrics**, not fully human-labeled facade ground truth.

## Problem Framing

Modern geospatial applications increasingly need more than isolated points,
polygons, and lines. Routing, search, delivery, accessibility, urban analytics,
augmented reality, field operations, and map-quality systems all depend on
semantic relationships between entities:

```text
this POI is inside or associated with this building
this business uses this address
this storefront faces this street
this sign or entrance supports this place candidate
this imagery observation may represent a missing map entity
```

Those relationships are often implicit, incomplete, or inconsistent across open
geospatial layers. Overture provides rich base layers, but upstream applications
still need reproducible ways to convert raw entity layers into structured,
reviewable semantic representations.

The repository is set up as a research baseline for that conversion. It
produces intermediate tables that make entity relationships explicit, assigns
evidence labels instead of opaque decisions, and separates high-confidence
links from review queues. The goal is to support upstream systems that need
application-ready geospatial semantics, including:

- POI-to-building and POI-to-address relationship graphs
- storefront and entrance localization for navigation and logistics
- semantic map enrichment from field imagery
- quality-assurance queues for human map review
- benchmark datasets for comparing conflation and facade-ranking methods

In this framing, the pipelines are not only solving local matching tasks. They
are constructing a bridge from raw geospatial geometry to semantic map entities
that downstream systems can reason over, audit, and improve.

## Repository Layout

```text
repo-root/
|-- Conflation_pipeline/          # Problem 1: POI -> address/building/street
|-- Problem2_facade_pipeline/     # Problem 2: building facade/storefront edge matching
|-- Problem3_seed_enrichment/     # Problem 3: imagery-seeded POI candidate generation
|-- Evaluation_pipeline/          # Cross-pipeline evaluation and review queues
|-- benchmark_dataset/            # Shared benchmark schema and candidate CSVs
|-- data/analysis/                # Lightweight committed analysis artifacts
|-- LICENSE
`-- README.md
```

Module-level documentation is available in:

- `Conflation_pipeline/README_GITHUB.md`
- `Problem2_facade_pipeline/README_GITHUB.md`
- `Problem3_seed_enrichment/README_GITHUB.md`
- `Evaluation_pipeline/README.md`
- `benchmark_dataset/README.md`

## High-Level Architecture

```text
Overture places, addresses, buildings, streets
        |
        v
Problem 1 conflation
        |
        | final_problem1_conflation.csv
        v
Problem 2 facade matching
        |
        | facade predictions + proxy evaluation rows
        v
Evaluation pipeline and benchmark review queues

Field imagery observations
        |
        v
Problem 3 seed aggregation, local Overture check, licensed evidence check
        |
        v
review queue + proposed additions for human approval
```

Problem 2 depends on Problem 1 output. Problem 3 is related but can run as an
independent enrichment workflow.

## Setup

This project is Python-based. A virtual environment is recommended.

```bash
cd /path/to/repo-root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the dependencies for the pipeline you plan to run:

```bash
pip install -r Conflation_pipeline/requirements.txt
pip install -r Problem2_facade_pipeline/requirements.txt
pip install -r Problem3_seed_enrichment/requirements.txt
```

The evaluation pipeline uses only the Python standard library.

## Data Policy and Reproducibility Notes

Raw Overture extracts, generated outputs, virtual environments, caches, and
large local assets are intentionally ignored by Git. This keeps the repository
reviewable while allowing experiments to be reproduced from source scripts.

Important `.gitignore` behavior:

- `**/data/raw/` is ignored.
- `**/outputs/` is ignored.
- `*.geojson` and `*.parquet` are ignored by default.
- Data download/source scripts are kept visible under `data/` folders.
- Curated benchmark CSVs under `benchmark_dataset/` and
  `Problem2_facade_pipeline/benchmark_dataset/` are committed.

For local runs, place Overture extracts under the expected pipeline raw-data
folder, most commonly:

```text
Problem2_facade_pipeline/data/raw/
|-- places.geojson
|-- addresses.geojson
|-- buildings.geojson
`-- segments.geojson
```

The Problem 1 loader also checks its own raw folder:

```text
Conflation_pipeline/data/raw/
```

If files are missing, the loaders print the searched paths.

## Quick Start

Run Problem 1:

```bash
python Conflation_pipeline/src/run_pipeline.py
```

Run Problem 2 after Problem 1:

```bash
python Problem2_facade_pipeline/src/run_pipeline.py
```

Run the evaluation dashboard:

```bash
python3 Evaluation_pipeline/src/evaluate.py
```

Run Problem 3 in local-only seed verification mode:

```bash
python Problem3_seed_enrichment/src/run_pipeline.py
```

Run the Problem 3 workflow with an explicitly licensed evidence CSV:

```bash
python Problem3_seed_enrichment/src/run_pipeline.py \
  --evidence-csv /path/to/licensed_poi_evidence.csv
```

## Problem 1: POI, Address, Building, and Street Conflation

Problem 1 creates an interpretable relationship table:

```text
POI -> matched address -> matched building -> matched street
```

Core stages:

- load Overture places, addresses, buildings, and street segments
- normalize names and address text
- match POIs to candidate addresses
- match and validate building relationships
- classify final labels for reviewability
- add street context
- write summary and manual-evaluation templates

Primary source files:

```text
Conflation_pipeline/src/
|-- run_pipeline.py
|-- data_loader.py
|-- address_matcher.py
|-- building_matcher.py
|-- building_validator.py
|-- street_connector.py
|-- labeler.py
|-- normalization.py
`-- summary_writer.py
```

Generated outputs:

| Output | Purpose |
| --- | --- |
| `Conflation_pipeline/outputs/final_problem1_conflation.csv` | Final POI/address/building/street table. |
| `Conflation_pipeline/outputs/problem1_summary.csv` | Counts by final label, match method, POI class, building status, and address status. |
| `Conflation_pipeline/outputs/problem1_manual_eval_template.csv` | Sampled review template for human validation. |

Current documented run summary:

| Metric | Count |
| --- | ---: |
| Total POIs | 4,039 |
| `same_building_confirmed` | 2,798 |
| `text_and_distance_match` | 699 |
| `needs_review` | 442 |
| `non_building_poi` | 82 |
| `building_validated_candidate` | 18 |

These labels are evidence categories from the rule-based pipeline. They should
not be presented as human-verified real-world accuracy.

## Problem 2: Facade / Storefront Matching

Problem 2 takes building-associated POIs from Problem 1 and chooses a likely
facade edge from the matched building footprint. This is useful for storefront
localization, sign placement, entrance routing, delivery navigation, and
map-quality review.

The pipeline includes:

- facade candidate generation from building polygons
- nearest-edge baseline
- street-aware reranking
- tuned reranker with distance guardrails
- strict and permissive proxy evaluation modes
- non-building POI filtering
- city-level, shared-building, and corner-ambiguity analyses
- failure-analysis CSV outputs

Primary source files:

```text
Problem2_facade_pipeline/src/
|-- run_pipeline.py
|-- config.py
|-- overture_proxy_data_builder.py
|-- proxy_benchmark_builder.py
|-- proxy_facade_evaluator.py
`-- tune_facade_reranker.py
```

### Proxy Benchmark

The benchmark uses field-observed sign, window-sign, and entrance/front-door
points for Boulder and Louisville, Colorado. These observations provide
storefront-related point evidence, but they do not contain explicit verified
facade IDs. The evaluation derives a proxy facade label by assigning each
sign/entrance point to the nearest candidate building edge.

This means the reported metric is:

```text
pipeline-selected facade edge == proxy nearest edge from observed sign/door point
```

It is not the same as human-reviewed facade correctness.

### Current Problem 2 Results

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

The corner-tolerant metric is diagnostic only. The official strict proxy
agreement remains 54.6667%.

### Useful Problem 2 Commands

Build proxy-area Overture extracts:

```bash
python Problem2_facade_pipeline/src/overture_proxy_data_builder.py --overwrite
```

Run strict/permissive building-only evaluation:

```bash
USE_PROXY_AREA_OVERTURE_DATA=1 \
PROXY_EVAL_INCLUDE_NEEDS_REVIEW=1 \
EXCLUDE_NON_BUILDING_POIS_FROM_PROXY_EVAL=1 \
python Conflation_pipeline/src/run_pipeline.py

USE_PROXY_AREA_OVERTURE_DATA=1 \
PROXY_EVAL_INCLUDE_NEEDS_REVIEW=1 \
EXCLUDE_NON_BUILDING_POIS_FROM_PROXY_EVAL=1 \
python Problem2_facade_pipeline/src/run_pipeline.py
```

Tune the facade reranker:

```bash
python Problem2_facade_pipeline/src/tune_facade_reranker.py
```

Run with the tuned reranker:

```bash
TUNE_FACADE_RERANKER=1 python Problem2_facade_pipeline/src/run_pipeline.py
```

Experimental modes are available but disabled by default:

```bash
ENABLE_LOCAL_STOREFRONT_HEURISTICS=1 python Problem2_facade_pipeline/src/run_pipeline.py
ENABLE_SHARED_BUILDING_FACADE_LOGIC=1 python Problem2_facade_pipeline/src/run_pipeline.py
```

## Problem 3: Imagery-Seeded POI Verification and Enrichment

Problem 3 converts street-level sign/entrance observations into reviewable POI
addition candidates. It is intentionally conservative: an observed sign is a
seed, not enough evidence by itself to publish a new POI.

Workflow:

```text
sign / entrance observations
-> aggregate observations into one seed per business
-> check the local Overture place extract
-> optionally join explicitly licensed external POI evidence
-> score evidence and write a review queue
-> emit proposed additions only for externally verified, locally absent POIs
```

Primary source files:

```text
Problem3_seed_enrichment/src/
|-- run_pipeline.py
|-- seed_builder.py
|-- local_overture_matcher.py
|-- licensed_evidence_matcher.py
|-- raw_image_extractor.py
|-- text_utils.py
`-- config.py
```

Problem 3 can operate on annotated Zephr-style CSVs or on raw image folders.
Raw-image mode uses local Apple Vision OCR through `tools/vision_ocr.m` and
records extraction uncertainty in review queues.

Generated outputs:

| Output | Purpose |
| --- | --- |
| `imagery_poi_seeds.csv` | One normalized seed per observed business. |
| `imagery_poi_enriched_candidates.csv` | Local and external evidence table. |
| `imagery_poi_review_queue.csv` | Rows needing human decision. |
| `proposed_overture_additions.csv` | Locally absent, externally verified candidates only. |
| `problem3_summary.csv` | Counts by workflow status. |
| `raw_image_extractions.csv` | OCR, EXIF, filename-hint, and acceptance audit table. |
| `raw_image_extraction_review_queue.csv` | Images withheld until extraction issues are reviewed. |

Important provenance boundary: Google Places content is deliberately not wired
into the export workflow. External enrichment must come from a source whose
license and provenance support the intended downstream contribution workflow.

## Evaluation Pipeline

The evaluation pipeline reports different evidence levels separately:

| Metric type | Meaning |
| --- | --- |
| Problem 1 internal quality | Automated coverage and consistency checks, not verified accuracy. |
| Problem 1 human-reviewed pilot accuracy | Accuracy only from completed human reviewer verdicts. |
| Problem 1 LLM-assisted silver estimate | Optional research-assisted estimate, not golden-set accuracy. |
| Problem 2 facade proxy agreement | Agreement with sign/entrance proxy labels, not human-verified facade accuracy. |

Run:

```bash
python3 Evaluation_pipeline/src/evaluate.py
```

Optional LLM-assisted silver input:

```bash
python3 Evaluation_pipeline/src/evaluate.py \
  --llm-silver-input /path/to/problem1_review_queue_llm_verified.csv
```

Generated outputs:

| Output | Purpose |
| --- | --- |
| `evaluation_summary.csv` | Concise dashboard for demos and progress tracking. |
| `problem1_quality_metrics.csv` | Full automated Problem 1 quality report. |
| `problem1_automated_flags.csv` | Suspicious or review-worthy POIs. |
| `problem1_review_queue.csv` | Balanced challenge sample for failure analysis. |
| `problem1_random_review_queue.csv` | Random sample for estimating overall performance. |
| `problem1_reviewed_metrics.csv` | Human-reviewed pilot metrics. |
| `problem1_llm_silver_metrics.csv` | Optional LLM-assisted estimates. |
| `problem2_proxy_metrics.csv` | Problem 2 proxy agreement metrics. |

## Benchmark Dataset

`benchmark_dataset/` contains a semi-automatic review framework for future
manual evaluation of both Problem 1 and Problem 2. It is not complete ground
truth until reviewer labels are filled.

Key files:

| File | Purpose |
| --- | --- |
| `benchmark_candidates.csv` | Balanced candidate set for review. |
| `reviewed_ground_truth.csv` | Reviewer-label target file. |
| `benchmark_statistics.csv` | Candidate distribution summary. |
| `benchmark_schema.md` | Column definitions and reviewer-label guidance. |
| `dataset_inspection_report.md` | Inspection of the Boulder/Louisville sign and entrance reference data. |

Generate or refresh candidates:

```bash
python benchmark_dataset/generate_benchmark_candidates.py
```

Recommended reviewer labels:

| Label | Meaning |
| --- | --- |
| `correct` | The pipeline relationship appears correct. |
| `incorrect` | The pipeline relationship appears wrong. |
| `ambiguous` | Multiple relationships are plausible. |
| `not_applicable` | The POI should not be evaluated for this relationship. |
| `insufficient_evidence` | Available evidence is not enough to decide. |

## Design Choices

- **Interpretable baselines first.** The pipelines are rule-based and designed
  to expose evidence, not hide it behind a black-box score.
- **Reviewability over automatic publication.** Outputs include review queues,
  summaries, and diagnostic buckets.
- **Strict metric labeling.** Proxy agreement, internal quality, silver labels,
  and human-reviewed accuracy are kept separate.
- **Conservative enrichment.** Problem 3 does not publish imagery-discovered
  POIs directly; it creates provenance-aware candidates for human approval.
- **Failure analysis as a first-class output.** Shared buildings, long facades,
  corner ambiguity, noisy proxy points, and non-building POIs are explicitly
  surfaced rather than smoothed over.

## Known Limitations

- Problem 1 is a rule-based baseline and still requires manual review for true
  accuracy measurement.
- Problem 2 evaluates against derived sign/entrance proxy labels, not a fully
  human-labeled facade dataset.
- Overture building footprints can be too coarse for individual storefronts,
  especially in malls, plazas, and multi-tenant buildings.
- Long building edges may contain several businesses with no explicit
  storefront segmentation.
- Street and orientation signals can hurt when they overpower distance-based
  evidence.
- Problem 3 requires explicitly licensed external evidence before emitting
  proposed additions.

## Future Work

- Build a human-labeled golden dataset for POI-address-building-street
  relationships.
- Add verified facade IDs for a small but carefully reviewed storefront set.
- Generate visual review packets that combine photos, sign/door points,
  building footprints, predicted facades, and reviewer decisions.
- Improve shared-building and long-facade segmentation.
- Model separate behavior for confirmed Problem 1 rows versus `needs_review`
  rows.
- Expand beyond Boulder and Louisville, Colorado to measure regional error
  patterns.
- Add CI checks for deterministic pipeline runs on a small fixture dataset.

## License

This repository is licensed under the Apache License 2.0. See `LICENSE`.
