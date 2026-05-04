# This file extracts the weakest POI examples for manual inspection.
#
# Goal:
# After scoring and failure analysis, we want a smaller CSV that only contains
# the POIs that still look genuinely problematic.
#
# Important detail:
# We do NOT want natural or non-building POIs (mountains, lakes, trails, glaciers, etc.)
# to show up here just because they are missing building/address information.
#
# So this file:
# 1. reads the failure-analysis output
# 2. determines whether each POI is building-based or non-building
# 3. keeps only rows that are suspicious / need review AND are building-based
# 4. saves a smaller CSV for manual inspection

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Helper: decide whether a POI is building-based
# This should match the same logic used in the scoring files
# ---------------------------------------------------------
def is_building_type(types_value):
    """
    Returns True if the POI is likely associated with a building.
    Returns False for natural / geographic / large open-area features.
    """
    types = str(types_value).lower()

    non_building_keywords = [
        "natural",
        "peak",
        "mountain",
        "water",
        "lake",
        "park",
        "trail",
        "forest",
        "river",
        "glacier",
        "open_space",
        "hiking",
        "historic_marker",
    ]

    for keyword in non_building_keywords:
        if keyword in types:
            return False

    return True


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    input_csv = BASE_DIR / "outputs" / "poi_failure_analysis.csv"
    output_csv = BASE_DIR / "outputs" / "review_examples.csv"

    # Load the analyzed POI table
    df = pd.read_csv(input_csv)

    # Recompute whether each POI should be treated as building-based
    df["is_building_poi"] = df["types"].apply(is_building_type)

    # Keep rows that still look weak in scoring or failure analysis
    review_df = df[
        (df["confidence_label"] == "suspicious") |
        (df["review_label"] == "needs_review")
    ].copy()

    # Remove non-building POIs from manual review
    # because they are expected to lack building/address/entrance info
    review_df = review_df[review_df["is_building_poi"] == True].copy()

    # Sort weakest examples first
    review_df = review_df.sort_values(
        by=["conflation_score", "review_label", "name"],
        ascending=[True, True, True]
    )

    # Keep the most useful columns for inspection
    keep_cols = [
        "name",
        "types",
        "address",
        "overture_building_id",
        "entrance_type",
        "entrance_street",
        "osm_confirmed",
        "address_street_mismatch",
        "co_tenants",
        "building_tenants",
        "conflation_score",
        "confidence_label",
        "failure_modes",
        "review_label",
        "is_building_poi",
    ]

    review_df = review_df[keep_cols]

    # Preview first few rows
    print(review_df.head(20))
    print("\nTotal rows needing inspection:", len(review_df))

    # Save review file
    review_df.to_csv(output_csv, index=False)
    print(f"\nSaved review examples to: {output_csv}")