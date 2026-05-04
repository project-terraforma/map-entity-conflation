# This file finds POIs that are expected to have a building
# but are missing a building link.
#
# Why this matters:
# This is one of the strongest real error sets in the project.
# A restaurant, store, contractor, or service business usually
# should have a building/address/entrance relationship.
#
# So this file:
# 1. reads the expected-building analysis output
# 2. keeps only POIs labeled "expected_building"
# 3. filters to rows missing a building link
# 4. keeps useful columns for inspection
# 5. saves a focused CSV of likely conflation failures

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Helper: treat NaN / empty strings / "none" as missing
# ---------------------------------------------------------
def is_missing(value):
    return pd.isna(value) or str(value).strip() == "" or str(value).strip().lower() == "none"


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    input_csv = BASE_DIR / "outputs" / "poi_expected_building.csv"
    output_csv = BASE_DIR / "outputs" / "business_missing_building.csv"
    summary_csv = BASE_DIR / "outputs" / "business_missing_building_summary.csv"

    # Load the expected-building analysis results
    df = pd.read_csv(input_csv)

    # Keep only POIs that we believe should map to a building
    expected_df = df[df["expected_building_label"] == "expected_building"].copy()

    # From those, keep only rows that are missing a building link
    missing_building_df = expected_df[
        expected_df["overture_building_id"].apply(is_missing)
    ].copy()

    # Add a simple priority label to make manual review easier
    # High priority = expected building + suspicious
    # Medium priority = expected building + needs review
    # Lower priority = everything else in this filtered set
    def assign_priority(row):
        if row.get("confidence_label") == "suspicious":
            return "high_priority"
        elif row.get("review_label") == "needs_review":
            return "medium_priority"
        else:
            return "lower_priority"

    missing_building_df["review_priority"] = missing_building_df.apply(assign_priority, axis=1)

    # Sort most important cases first
    missing_building_df = missing_building_df.sort_values(
        by=["review_priority", "conflation_score", "name"],
        ascending=[True, True, True]
    )

    # Keep the most useful columns for manual inspection
    keep_cols = [
        "name",
        "types",
        "address",
        "overture_building_id",
        "entrance_lat",
        "entrance_lon",
        "entrance_type",
        "entrance_street",
        "osm_confirmed",
        "address_street_mismatch",
        "conflation_score",
        "confidence_label",
        "failure_modes",
        "review_label",
        "expected_building_label",
        "expected_building_reason",
        "review_priority",
    ]

    missing_building_df = missing_building_df[keep_cols]

    # Preview first few rows
    print(missing_building_df.head(20))
    print("\nTotal expected-building POIs missing a building link:", len(missing_building_df))

    print("\nPriority counts:")
    print(missing_building_df["review_priority"].value_counts())

    # Save summary
    summary = missing_building_df["review_priority"].value_counts().reset_index()
    summary.columns = ["review_priority", "count"]

    # Save outputs
    missing_building_df.to_csv(output_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"\nSaved focused error set to: {output_csv}")
    print(f"Saved summary to: {summary_csv}")