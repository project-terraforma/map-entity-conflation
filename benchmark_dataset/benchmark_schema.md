# Benchmark Schema

This benchmark is a semi-automatic review framework for future evaluation of:

- Problem 1: POI to address, building, and street conflation
- Problem 2: POI to facade/storefront matching

It is not ground truth until a human reviewer fills `reviewer_label` and `reviewer_notes`.

## Candidate Columns

| Column | Description |
| --- | --- |
| `poi_id` | Overture place id or pipeline POI id. |
| `poi_name` | POI name when available. |
| `building_id` | Matched building id from the available pipeline output. |
| `matched_address` | Problem 1 matched address text when available. Blank if Problem 1 output is absent. |
| `street_name` | Matched or nearest street name when available. Blank if unavailable. |
| `facade_id` | Problem 2 facade id, generated as building id plus facade index. |
| `confidence_label` | Pipeline confidence/review label. |
| `ambiguity_bucket` | Coarse review bucket derived from existing labels/evidence. |
| `region` | Benchmark region inferred from POI coordinates. |
| `benchmark_type` | Sampling bucket used to include a balanced review set. |
| `reviewer_label` | Empty field for future human judgment. |
| `reviewer_notes` | Empty field for reviewer explanation or uncertainty. |

## Benchmark Types

| Type | Meaning |
| --- | --- |
| `strong_match` | High-confidence facade match candidate. |
| `ambiguous_match` | Candidate from the multiple-close-facades review bucket. |
| `corner_or_close_facade_case` | Candidate where facade ambiguity is explicitly driven by close facade distances. |
| `multi_tenant_or_shared_facade` | Candidate sharing a facade/building context with other POIs. |
| `no_building_match_failure` | Candidate with no usable matched building. |
| `likely_nonbuilding_poi` | Candidate whose existing Overture category fields suggest a non-building POI. |

## Acceptable Reviewer Labels

Reviewer labels should be added only after manual inspection.

Recommended labels:

| Reviewer Label | Use When |
| --- | --- |
| `correct` | The pipeline relationship appears correct. |
| `incorrect` | The pipeline relationship appears wrong. |
| `ambiguous` | Multiple relationships are plausible from available evidence. |
| `not_applicable` | The POI should not be evaluated for this relationship, such as some outdoor or non-building POIs. |
| `insufficient_evidence` | The reviewer cannot decide from available data. |

For Problem 1, the reviewer should evaluate POI-address-building-street consistency.

For Problem 2, the reviewer should evaluate whether the selected facade/storefront edge is plausible.

## Ambiguity Handling

Ambiguity is not automatically incorrect. A POI can be ambiguous because:

- it lies near a building corner
- the building has many similarly distant facade edges
- multiple tenants share one building
- the POI point is placed inside or near a large footprint
- no entrance geometry is available

Reviewers should use `ambiguous` when the candidate cannot be confidently marked correct or incorrect.

## Problem 1 vs Problem 2 Evaluation

Problem 1 evaluates relationships among:

- POI
- address
- building
- street

Problem 2 evaluates:

- POI
- matched building
- selected facade edge
- optional street context

The same benchmark row can support both evaluations when all fields are available. In the current local workspace, Problem 1 final output is not present, so Problem 1-specific fields may be blank.
