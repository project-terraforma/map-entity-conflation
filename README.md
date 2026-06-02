# Overture Map Entity Conflation

This repository contains three related Overture / Project Terraforma pipelines:

- `Conflation_pipeline/`: Problem 1, linking POIs to addresses, buildings, and streets.
- `Problem2_facade_pipeline/`: Problem 2, using Problem 1 output to match valid building-based POIs to likely building facade edges.
- `Problem3_seed_enrichment/`: Problem 3, verifying imagery-discovered POI seeds and producing reviewable enriched additions.

Problem 2 depends on Problem 1. Run Problem 1 first, then run Problem 2.

```bash
python Conflation_pipeline/src/run_pipeline.py
python Problem2_facade_pipeline/src/run_pipeline.py
```

Technical docs:

- `Conflation_pipeline/README_GITHUB.md`
- `Problem2_facade_pipeline/README_GITHUB.md`
- `Problem3_seed_enrichment/README_GITHUB.md`

Raw data and generated outputs are ignored by git. Place local Overture extracts under the pipeline `data/raw/` folders when running locally.
