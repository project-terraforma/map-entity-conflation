# Overture POI Conflation Baseline

This project builds a baseline analysis pipeline for understanding missing and weak POI-to-building links in preprocessed Overture-style POI data.

The starting point for this work is the `overture-poi-preprocessing` dataset, which provides corrected POI entrance coordinates and related metadata. On top of that, this project adds a structured analysis pipeline to answer questions like:

- Which POIs appear to have strong building linkage evidence?
- Which POIs look suspicious or ambiguous?
- Which POIs should reasonably map to buildings but currently do not?
- What patterns explain those missing building links?

The goal is not to build a full production conflation system yet. Instead, this repo creates a strong **baseline diagnostic pipeline** that helps identify and categorize likely conflation failures.

---

## Project Motivation

In real-world map data, POIs, buildings, addresses, and streets are tightly connected.

Examples:
- A restaurant should usually be inside or attached to a building.
- That building should usually have an address.
- The address should relate to the correct street.
- The POI location should ideally be closer to the entrance than to a rough centroid.

However, these relationships are often incomplete or inconsistent in raw map data. A POI may:
- have a name but no building link,
- have an entrance point but no building assignment,
- have an address that does not align with the street,
- or represent a natural feature where a building link should not exist at all.

This project helps separate those cases and explain them systematically.

---

## Repository Goal

This repo builds a step-by-step analysis pipeline that:

1. loads and flattens POI tile JSON data,
2. assigns baseline confidence scores,
3. diagnoses broad failure modes,
4. estimates whether a POI is expected to have a building,
5. isolates likely true missing-building errors,
6. categorizes why those missing-building cases may be happening.

This gives a much more focused view of conflation issues than simply listing all POIs with weak metadata.

---

## Dataset Source

This project uses preprocessed POI tile files from:

`overture-poi-preprocessing`

That dataset includes useful fields such as:
- POI name and type
- original coordinates
- corrected entrance coordinates
- `overture_building_id`
- address
- entrance-related metadata
- context fields such as street mismatch and tenant information

This repo assumes those JSON tile files are available under:

```text
overture-poi-preprocessing/z14/

Folder Structure
V2_overture_conflation/
├── overture-poi-preprocessing/
│   └── z14/
├── outputs/
├── src/
│   ├── load_tiles.py
│   ├── baseline_conflation_score.py
│   ├── failure_analysis.py
│   ├── review_examples.py
│   ├── expected_building_analysis.py
│   ├── business_missing_building.py
│   └── candidate_fix_rules.py
└── README.md
Pipeline Overview

The pipeline is organized into several analysis stages.

1. load_tiles.py

This file is the data ingestion and flattening step.

It:

opens every JSON tile file,
reads the waypoints list inside each file,
extracts key fields,
converts them into row-based records,
combines everything into one pandas DataFrame,
saves the result as poi_waypoints_flat.csv.

This stage turns many nested JSON files into one flat table that is much easier to analyze.

2. baseline_conflation_score.py

This file assigns a baseline confidence score to each POI.

It uses simple heuristic rules based on:

building linkage,
entrance coordinates,
entrance type,
address presence,
address/street mismatch,
OSM confirmation.

It also includes logic to avoid unfairly penalizing obvious non-building POIs like lakes, parks, and mountains.

Outputs:

poi_scored.csv
poi_score_summary.csv
3. failure_analysis.py

This file assigns broader failure modes and review labels.

It helps answer:

Why does this POI look weak?
Is it missing a building link?
Is it missing an address?
Is it likely a non-building POI?
Is it a multi-tenant ambiguity case?

Outputs:

poi_failure_analysis.csv
poi_failure_summary.csv
4. review_examples.py

This file extracts rows that still look problematic after scoring and failure analysis.

It creates a smaller inspection set for manual review, focused on likely building-related POIs.

Output:

review_examples.csv
5. expected_building_analysis.py

This file classifies POIs into:

expected_building
not_expected_building
uncertain

It uses both:

the types field,
and the POI name.

This is important because not all POIs should be expected to have building links.

Outputs:

poi_expected_building.csv
poi_expected_building_summary.csv
6. business_missing_building.py

This file creates a focused error set.

It keeps only POIs that:

are classified as expected_building,
but are missing an overture_building_id.

This is one of the strongest candidate sets for real conflation failures.

Outputs:

business_missing_building.csv
business_missing_building_summary.csv
7. candidate_fix_rules.py

This file analyzes the focused error set and assigns more specific diagnostic buckets.

For example, it can identify cases such as:

entrance geometry exists but entrance type is missing,
all core metadata is missing,
special landmark cases,
address-only cases without entrance geometry.

Outputs:

candidate_fix_rules.csv
candidate_fix_rules_summary.csv
Current Findings

From the current analysis:

The flattened dataset contains 44,664 POIs.
A large portion of POIs are medium-confidence under the baseline scoring system.
A smaller subset is suspicious and worth review.
After type-aware filtering, many natural features are correctly excluded from building-related error analysis.
Among POIs classified as expected_building, 1,440 are missing a building link.
In the current diagnostic breakdown, these cases strongly suggest that many POIs already have entrance geometry but are missing the metadata needed to complete building assignment.

This is an important result because it suggests the problem is often incomplete enrichment, not total absence of spatial evidence.

Why Type Awareness Matters

One major challenge in POI conflation is that not all POIs should be treated with the same rules.

Examples:

A restaurant should usually have a building link.
A clothing store should usually have an address and entrance.
A glacier or mountain should not be penalized for missing a building.

Without type-aware logic, natural features appear as false positives in the error set.

This project explicitly addresses that by separating:

building-like POIs,
non-building POIs,
uncertain cases.
Installation

Create and activate a Python environment if desired, then install pandas:

pip install pandas

If you use PowerShell on Windows:

pip install pandas
How to Run

Run the scripts from the src/ folder or from the project root.

Suggested order:

1. Flatten tile files
python src/load_tiles.py
2. Score POIs
python src/baseline_conflation_score.py
3. Run failure analysis
python src/failure_analysis.py
4. Extract review examples
python src/review_examples.py
5. Classify expected-building POIs
python src/expected_building_analysis.py
6. Extract expected-building POIs missing building links
python src/business_missing_building.py
7. Diagnose focused error set
python src/candidate_fix_rules.py
Main Output Files
Core outputs
outputs/poi_waypoints_flat.csv
outputs/poi_scored.csv
outputs/poi_failure_analysis.csv
Review / focused analysis outputs
outputs/review_examples.csv
outputs/poi_expected_building.csv
outputs/business_missing_building.csv
outputs/candidate_fix_rules.csv
Summary outputs
outputs/poi_score_summary.csv
outputs/poi_failure_summary.csv
outputs/poi_expected_building_summary.csv
outputs/business_missing_building_summary.csv
outputs/candidate_fix_rules_summary.csv
Limitations

This project is still a baseline analysis pipeline, not a full conflation engine.

Current limitations include:

heuristic scoring instead of learned scoring,
keyword-based type classification,
no direct building polygon search yet,
no actual reassignment of missing building links yet,
no spatial nearest-building recovery logic yet,
some broad categories remain uncertain.

So the current system is strongest as a diagnostic and prioritization tool.

Next Step

The next major step is to move from diagnosis to recovery.

Planned next stage:

use entrance coordinates to search for likely candidate buildings,
test whether the entrance point falls inside a building polygon,
otherwise find the nearest building,
propose candidate building assignments for POIs currently missing overture_building_id.

That would turn this project from:

“finding and explaining likely conflation failures”

into:

“proposing concrete fixes for those failures.”
Summary

This project builds a structured baseline pipeline for POI conflation analysis on top of preprocessed Overture POI data.

It helps answer:

which POIs look reliable,
which ones look suspicious,
which ones should probably have building links,
and what kind of metadata or conflation gap may be preventing those links.

This provides a strong foundation for future work on:

candidate generation,
building assignment recovery,
improved type classification,
and full POI ↔ building ↔ address ↔ street conflation.

---

## A couple optional sections you could add later
You could also add:
- **Example outputs**
- **Screenshots / sample CSV snippets**
- **Future work**
- **Acknowledgments / dataset credit**

## Next move for option 2
The next coding step should be a file that starts trying to **recover candidate buildings from entrance coordinates**. A good next filename would be:

```text
recover_building_candidates.py