# This file estimates whether a POI should reasonably be expected
# to have building/address/entrance information.
#
# Why this matters:
# Some POIs are businesses or services and should probably map to a building.
# Others are natural features or open-area places, where missing building data is normal.
#
# The earlier building/non-building rule was too simple.
# This file improves that by using both:
# 1. the POI type
# 2. the POI name
#
# Output:
# - an expected-building label for each POI
# - a reason explaining why that label was assigned
# - a summary CSV

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Helper: classify whether a POI should be expected to map
# to a building
# ---------------------------------------------------------
def classify_expected_building(row):
    name = str(row.get("name", "")).lower()
    types = str(row.get("types", "")).lower()

    # Strong building-type categories
    building_type_keywords = [
        "restaurant",
        "grill",
        "cafe",
        "store",
        "shop",
        "clothing",
        "contractor",
        "real_estate",
        "service",
        "repair",
        "organization",
        "nutritionist",
        "taxidermist",
        "jewelry",
        "religious_organization",
        "computer",
        "desserts",
        "print_media",
        "landmark_and_historical_building",
        "college_university",
        "pet_breeder",
    ]

    # Strong non-building clues in types
    non_building_type_keywords = [
        "mountain",
        "peak",
        "lake",
        "river",
        "water",
        "glacier",
        "trail",
        "forest",
        "park",
        "structure_and_geography",
        "open_space",
        "hiking",
    ]

    # Strong non-building clues in names
    non_building_name_keywords = [
        "peak",
        "mountain",
        "mt.",
        "lake",
        "river",
        "glacier",
        "trail",
        "park",
        "open space",
        "creek",
        "canyon",
        "ridge",
        "hill",
        "falls",
    ]

    # Strong building clues in names
    building_name_keywords = [
        "grill",
        "cafe",
        "restaurant",
        "services",
        "service",
        "repair",
        "real estate",
        "advisor",
        "court",
        "village",
        "shop",
        "boutique",
        "store",
        "church",
        "ministries",
    ]

    # Rule 1: obvious non-building from type
    for keyword in non_building_type_keywords:
        if keyword in types:
            return "not_expected_building", f"type:{keyword}"

    # Rule 2: obvious non-building from name
    for keyword in non_building_name_keywords:
        if keyword in name:
            return "not_expected_building", f"name:{keyword}"

    # Rule 3: obvious building from type
    for keyword in building_type_keywords:
        if keyword in types:
            return "expected_building", f"type:{keyword}"

    # Rule 4: obvious building from name
    for keyword in building_name_keywords:
        if keyword in name:
            return "expected_building", f"name:{keyword}"

    # Rule 5: broad/unclear categories remain uncertain
    return "uncertain", "no_strong_signal"


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    input_csv = BASE_DIR / "outputs" / "poi_failure_analysis.csv"
    output_csv = BASE_DIR / "outputs" / "poi_expected_building.csv"
    summary_csv = BASE_DIR / "outputs" / "poi_expected_building_summary.csv"

    # Load prior output
    df = pd.read_csv(input_csv)

    # Apply classifier to every row
    results = df.apply(classify_expected_building, axis=1, result_type="expand")
    results.columns = ["expected_building_label", "expected_building_reason"]

    # Merge results back in
    df = pd.concat([df, results], axis=1)

    # Preview first few rows
    print(df[[
        "name",
        "types",
        "expected_building_label",
        "expected_building_reason"
    ]].head(20))

    print("\nExpected building counts:")
    print(df["expected_building_label"].value_counts())

    # Save summary
    summary = df["expected_building_label"].value_counts().reset_index()
    summary.columns = ["expected_building_label", "count"]

    # Save outputs
    df.to_csv(output_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"\nSaved expected-building analysis to: {output_csv}")
    print(f"Saved expected-building summary to: {summary_csv}")