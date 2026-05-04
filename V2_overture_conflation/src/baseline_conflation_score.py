# This file assigns a baseline confidence score to each POI.
#
# Key idea:
# Not all POIs should be treated the same.
# Some POIs (restaurants, shops) are building-based.
# Others (mountains, lakes, parks) are NOT.
#
# So this file:
# 1. Reads the flattened POI CSV
# 2. Determines if each POI is building-based or not
# 3. Applies different scoring rules based on POI type
# 4. Produces:
#    - a numeric conflation score
#    - a list of reasons explaining the score
#    - a confidence label (high / medium / low / suspicious)
# 5. Saves results to CSV

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# Helper function: classify whether a POI is building-based
# ---------------------------------------------------------
def is_building_poi(row):
    """
    Returns True if the POI is likely associated with a building.
    Returns False for natural / large geographic features.
    """

    types = str(row.get("types")).lower()

    # Keywords that indicate NON-building POIs
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
# Core scoring function
# ---------------------------------------------------------
def score_row(row):
    score = 0
    reasons = []

    # Determine if this POI should follow building rules
    is_building = is_building_poi(row)

    # -----------------------------------------------------
    # 1. Building linkage (only matters for building POIs)
    # -----------------------------------------------------
    if is_building:
        if pd.notna(row.get("overture_building_id")):
            score += 4
            reasons.append("has_building_id")
        else:
            score -= 3
            reasons.append("missing_building_id")
    else:
        # For natural features, building link is not expected
        reasons.append("non_building_poi")

    # -----------------------------------------------------
    # 2. Entrance point (only matters for building POIs)
    # -----------------------------------------------------
    if is_building:
        if pd.notna(row.get("entrance_lat")) and pd.notna(row.get("entrance_lon")):
            score += 2
            reasons.append("has_entrance_point")
        else:
            score -= 2
            reasons.append("missing_entrance_point")
    else:
        reasons.append("entrance_not_expected")

    # -----------------------------------------------------
    # 3. Entrance type (only relevant for building POIs)
    # -----------------------------------------------------
    entrance_type = row.get("entrance_type")

    if is_building:
        if isinstance(entrance_type, str):
            entrance_type = entrance_type.lower()

            if entrance_type == "facade":
                score += 2
                reasons.append("facade_entrance")

            elif entrance_type == "entrance":
                score += 1
                reasons.append("generic_entrance")

            elif entrance_type == "center":
                score -= 3
                reasons.append("center_fallback")

            else:
                score -= 1
                reasons.append(f"unusual_entrance_type_{entrance_type}")
        else:
            score -= 1
            reasons.append("missing_entrance_type")
    else:
        reasons.append("entrance_type_not_applicable")

    # -----------------------------------------------------
    # 4. Address (only relevant for building POIs)
    # -----------------------------------------------------
    if is_building:
        if pd.notna(row.get("address")) and str(row.get("address")).strip():
            score += 1
            reasons.append("has_address")
        else:
            score -= 2
            reasons.append("missing_address")
    else:
        reasons.append("address_not_expected")

    # -----------------------------------------------------
    # 5. Address-street mismatch
    # Still useful even outside strict building POIs
    # -----------------------------------------------------
    mismatch = row.get("address_street_mismatch")

    if str(mismatch).strip().lower() == "true":
        score -= 4
        reasons.append("address_street_mismatch")

    elif str(mismatch).strip().lower() == "false":
        score += 1
        reasons.append("street_consistent")

    else:
        reasons.append("street_mismatch_unknown")

    # -----------------------------------------------------
    # 6. OSM confirmation
    # Small positive signal if present
    # -----------------------------------------------------
    osm_confirmed = row.get("osm_confirmed")

    if str(osm_confirmed).strip().lower() == "true":
        score += 1
        reasons.append("osm_confirmed")

    elif str(osm_confirmed).strip().lower() == "false":
        reasons.append("not_osm_confirmed")

    else:
        reasons.append("osm_unknown")

    # Return score + explanation
    return score, ";".join(reasons)


# ---------------------------------------------------------
# Convert numeric score into label
# ---------------------------------------------------------
def classify_score(score):
    if score >= 7:
        return "high_confidence"
    elif score >= 3:
        return "medium_confidence"
    elif score >= 0:
        return "low_confidence"
    else:
        return "suspicious"


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    input_csv = BASE_DIR / "outputs" / "poi_waypoints_flat.csv"
    output_csv = BASE_DIR / "outputs" / "poi_scored.csv"
    summary_csv = BASE_DIR / "outputs" / "poi_score_summary.csv"

    # Load flattened POI data
    df = pd.read_csv(input_csv)

    # Apply scoring function to every row
    results = df.apply(score_row, axis=1, result_type="expand")
    results.columns = ["conflation_score", "score_reasons"]

    # Merge results back into original table
    df = pd.concat([df, results], axis=1)

    # Assign confidence label
    df["confidence_label"] = df["conflation_score"].apply(classify_score)

    # Preview
    print(df[["name", "conflation_score", "confidence_label", "score_reasons"]].head())
    print("\nConfidence label counts:")
    print(df["confidence_label"].value_counts())

    # Create summary table
    summary = df["confidence_label"].value_counts().reset_index()
    summary.columns = ["confidence_label", "count"]

    # Save outputs
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"\nSaved scored POIs to: {output_csv}")
    print(f"Saved summary to: {summary_csv}")