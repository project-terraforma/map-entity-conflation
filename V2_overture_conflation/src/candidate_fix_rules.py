# This file analyzes the focused error set:
# expected-building POIs that are missing a building link.
#
# Goal:
# Instead of only listing bad cases, we want to understand
# what kind of failure each case represents.
#
# This file:
# 1. reads business_missing_building.csv
# 2. assigns a likely failure bucket to each POI
# 3. creates a summary of failure buckets
# 4. saves the detailed output and summary
#
# This helps us answer:
# "Why is this expected-building POI missing a building link?"

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Helper: treat NaN / empty strings / "none" / "nan" as missing
# ---------------------------------------------------------
def is_missing(value):
    value_str = str(value).strip().lower()
    return pd.isna(value) or value_str == "" or value_str in {"none", "nan"}


# ---------------------------------------------------------
# Assign a likely failure bucket
# ---------------------------------------------------------
def assign_failure_bucket(row):
    # Read useful fields from the row
    address = row.get("address")
    entrance_lat = row.get("entrance_lat")
    entrance_lon = row.get("entrance_lon")
    entrance_type = row.get("entrance_type")
    osm_confirmed = str(row.get("osm_confirmed")).strip().lower()
    mismatch = str(row.get("address_street_mismatch")).strip().lower()
    types = str(row.get("types")).lower()

    # Presence/absence flags
    has_address = not is_missing(address)
    has_entrance_coords = not is_missing(entrance_lat) and not is_missing(entrance_lon)
    has_entrance_type = not is_missing(entrance_type)

    # Case 1:
    # Landmark / historical categories may need special handling.
    if "historical" in types or "landmark" in types:
        return "special_landmark_case"

    # Case 2:
    # No address, no entrance coordinates, and no entrance type.
    if not has_address and not has_entrance_coords and not has_entrance_type:
        return "business_missing_all_core_features"

    # Case 3:
    # Entrance coordinates exist, but entrance type is missing.
    if has_entrance_coords and not has_entrance_type:
        return "has_entrance_geometry_but_missing_entrance_type"

    # Case 4:
    # Entrance coordinates and entrance type both exist, but no building link.
    if has_entrance_coords and has_entrance_type:
        return "has_entrance_but_no_building_link"

    # Case 5:
    # Address exists, but entrance geometry is missing.
    if has_address and not has_entrance_coords:
        return "has_address_but_no_entrance"

    # Case 6:
    # Street mismatch.
    if mismatch == "true":
        return "address_street_conflict"

    # Case 7:
    # OSM-supported but still unlinked.
    if osm_confirmed == "true":
        return "osm_supported_but_unlinked"

    # -----------------------------------------------------
    # Default fallback
    # -----------------------------------------------------
    return "other_unclassified"


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    input_csv = BASE_DIR / "outputs" / "business_missing_building.csv"
    output_csv = BASE_DIR / "outputs" / "candidate_fix_rules.csv"
    summary_csv = BASE_DIR / "outputs" / "candidate_fix_rules_summary.csv"

    # Load the focused error set
    df = pd.read_csv(input_csv)

    # Optional debug example
    debug_row = df[df["name"] == "Aladdin Mediterranean Grill"].iloc[0]

    print("\n--- DEBUG: Aladdin Mediterranean Grill ---")
    print("address:", repr(debug_row.get("address")))
    print("entrance_lat:", repr(debug_row.get("entrance_lat")))
    print("entrance_lon:", repr(debug_row.get("entrance_lon")))
    print("entrance_type:", repr(debug_row.get("entrance_type")))
    print("osm_confirmed:", repr(debug_row.get("osm_confirmed")))
    print("address_street_mismatch:", repr(debug_row.get("address_street_mismatch")))
    print("------------------------------------------\n")

    # Assign a failure bucket to each row
    df["failure_bucket"] = df.apply(assign_failure_bucket, axis=1)

    # Preview first few rows
    print(df[[
        "name",
        "types",
        "address",
        "entrance_type",
        "osm_confirmed",
        "conflation_score",
        "failure_bucket"
    ]].head(20))

    print("\nFailure bucket counts:")
    print(df["failure_bucket"].value_counts())

    # Create summary table
    summary = df["failure_bucket"].value_counts().reset_index()
    summary.columns = ["failure_bucket", "count"]

    # Save outputs
    df.to_csv(output_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"\nSaved detailed bucketed output to: {output_csv}")
    print(f"Saved summary to: {summary_csv}")