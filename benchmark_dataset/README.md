# Shared Benchmark Dataset

This folder contains a semi-automatic benchmark framework for evaluating both repository tasks:

- Problem 1: POI to building, address, and street conflation
- Problem 2: POI to facade/storefront matching

The benchmark is designed for future manual review. It does not contain verified ground-truth labels yet.

## Current Status

The current local benchmark is generated primarily from Problem 2 outputs because the Problem 1 final output file is not present in this workspace.

Available generated inputs include:

- `outputs/problem2_facade_matches.csv`
- `outputs/problem2_ambiguity_analysis.csv`
- `outputs/problem2_ranker_comparison.csv`
- `outputs/problem2_nonbuilding_analysis.csv`
- `outputs/visualizations/captions.csv`

No Google Maps data is scraped or copied.

## Workflow

Run:

```bash
python benchmark_dataset/generate_benchmark_candidates.py
```

This creates or updates:

- `benchmark_candidates.csv`
- `reviewed_ground_truth.csv`
- `benchmark_statistics.csv`
- review packet folders under `review_packets/`

## Candidate Sampling

The generator builds a balanced candidate set from existing outputs:

- strong matches
- ambiguous matches
- corner or close-facade cases
- multi-tenant or shared-facade cases
- no-building-match failures
- likely non-building POIs

The generator leaves reviewer fields blank.

## Manual Review

Human reviewers should fill:

- `reviewer_label`
- `reviewer_notes`

Recommended reviewer labels are documented in `benchmark_schema.md`.

## Limitations

- This is not finalized ground truth.
- Problem 1 fields are blank unless `Conflation_pipeline/outputs/final_problem1_conflation.csv` exists.
- Problem 2 evaluation is currently based on confidence labels, ambiguity buckets, and visual review packets.
- Entrance data is not available locally.
- Multi-tenant storefront ambiguity remains difficult without verified facade labels.

## Future Goals

The benchmark should support comparison of:

- baseline Problem 1 and Problem 2 methods
- improved rule-based facade re-ranking
- future ML-based ranking models
- human-reviewed ground truth over the same candidate set
