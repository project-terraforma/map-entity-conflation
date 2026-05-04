# This file analyzes why a POI may be weak, ambiguous, or suspicious.
#
# Important idea:
# Not all POIs should be judged by building-based rules.
# Some POIs are businesses or services and should have building/address signals.
# Others are natural or large geographic features (mountains, lakes, parks, trails),
# where missing building/address/entrance info is normal.
#
# So this file:
# 1. reads the scored POI CSV
# 2. checks whether each POI is building-based or non-building
# 3. assigns likely failure modes
# 4. maps those failure modes into a broad review label
# 5. saves both the detailed results and a summary CSV

from pathlib import Path
import ast
import pandas as pd


# ---------------------------------------------------------
# Helper: treat NaN, empty strings, and "none" as missing
# ---------------------------------------------------------
def is_missing(value):
    return pd.isna(value) or str(value).strip() == "" or str(value).strip().lower() == "none"


# ---------------------------------------------------------
# Helper: decide whether this POI is building-based
# Same idea as the updated baseline scoring file
# ---------------------------------------------------------
def is_building_poi(row):
    """
    Returns True if the POI is likely associated with a building.
    Returns False for natural / large geographic features.
    """
    types = str(row.get("types")).lower()

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
    ]

    for keyword in non_building_keywords:
        if keyword in types:
            return False

    return True


# ---------------------------------------------------------
# Helper: safely parse list-like strings from CSV
# Example: "['A', 'B']" -> ['A', 'B']
# ---------------------------------------------------------
def parse_list_field(value):
    if is_missing(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []


# ---------------------------------------------------------
# Core function: assign detailed failure modes to one row
# ---------------------------------------------------------
def classify_failure(row):
    failures = []

    # Determine if this POI should be evaluated with building rules
    is_building = is_building_poi(row)

    # -----------------------------------------------------
    # Building-related checks only apply to building POIs
    # -----------------------------------------------------
    if is_building:
        # 1. Missing building link
        if is_missing(row.get("overture_building_id")):
            failures.append("missing_building_link")

        # 2. Missing address
        if is_missing(row.get("address")):
            failures.append("missing_address")

        # 3. Missing entrance point
        if is_missing(row.get("entrance_lat")) or is_missing(row.get("entrance_lon")):
            failures.append("missing_entrance_point")

        # 4. Weak or missing entrance type
        entrance_type = row.get("entrance_type")
        if is_missing(entrance_type):
            failures.append("missing_entrance_type")
        else:
            entrance_type = str(entrance_type).lower()

            if entrance_type == "center":
                failures.append("centroid_or_center_fallback")
            elif entrance_type not in ["facade", "entrance"]:
                failures.append(f"unusual_entrance_type:{entrance_type}")

    else:
        # For non-building POIs, missing building/address/entrance is normal
        failures.append("non_building_poi")

    # -----------------------------------------------------
    # Street mismatch can still be meaningful if it exists
    # -----------------------------------------------------
    mismatch = row.get("address_street_mismatch")
    if str(mismatch).strip().lower() == "true":
        failures.append("address_street_mismatch")

    # -----------------------------------------------------
    # Multi-tenant ambiguity only really matters for building POIs
    # -----------------------------------------------------
    if is_building:
        co_tenants = parse_list_field(row.get("co_tenants"))
        building_tenants = parse_list_field(row.get("building_tenants"))

        if len(co_tenants) > 1:
            failures.append("co_tenant_ambiguity")
        elif len(co_tenants) == 1:
            failures.append("co_tenant_present")

        if len(building_tenants) > 3:
            failures.append("large_multi_tenant_building")
        elif 1 <= len(building_tenants) <= 3:
            failures.append("building_tenants_present")

    # If nothing meaningful was flagged, mark as clean
    if not failures:
        failures.append("no_obvious_failure")

    return ";".join(failures)


# ---------------------------------------------------------
# Convert detailed failure modes into a broad review label
# ---------------------------------------------------------
def assign_review_label(failure_string):
    failures = failure_string.split(";")

    # Stronger red flags
    severe = {
        "missing_building_link",
        "address_street_mismatch",
        "missing_entrance_point",
        "centroid_or_center_fallback",
    }

    # Moderate ambiguity signals
    moderate = {
        "missing_address",
        "missing_entrance_type",
        "co_tenant_ambiguity",
        "large_multi_tenant_building",
        "co_tenant_present",
        "building_tenants_present",
    }

    # Non-building POIs should not be forced into review
    if "non_building_poi" in failures and len(failures) == 1:
        return "looks_ok"

    if any(f in severe for f in failures):
        return "needs_review"
    elif any(f in moderate for f in failures):
        return "possibly_ambiguous"
    else:
        return "looks_ok"


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Input and output paths
    input_csv = BASE_DIR / "outputs" / "poi_scored.csv"
    output_csv = BASE_DIR / "outputs" / "poi_failure_analysis.csv"
    summary_csv = BASE_DIR / "outputs" / "poi_failure_summary.csv"

    # Load the scored POI table
    df = pd.read_csv(input_csv)

    # Compute detailed failure modes and broad review labels
    df["failure_modes"] = df.apply(classify_failure, axis=1)
    df["review_label"] = df["failure_modes"].apply(assign_review_label)

    # Preview first few rows
    print(df[["name", "types", "confidence_label", "failure_modes", "review_label"]].head())

    print("\nReview label counts:")
    print(df["review_label"].value_counts())

    # Create summary table
    summary = df["review_label"].value_counts().reset_index()
    summary.columns = ["review_label", "count"]

    # Save outputs
    df.to_csv(output_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"\nSaved failure analysis to: {output_csv}")
    print(f"Saved failure summary to: {summary_csv}")