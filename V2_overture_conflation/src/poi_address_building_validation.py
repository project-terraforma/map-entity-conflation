# Poi_address_building_validation.py

import math
import pandas as pd

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
# Input = output from your POI -> address matching step
INPUT_FILE = r"..\outputs\poi_address_matches.csv"

# Output = building-validated final results
OUTPUT_FILE = r"..\outputs\poi_address_building_validated.csv"


# ------------------------------------------------------------
# Distance thresholds (in meters)
# ------------------------------------------------------------
# If matched address is very close to POI entrance,
# we treat it as likely belonging to the same building.
SAME_BUILDING_DISTANCE_M = 30.0

# If it is farther away but still somewhat nearby,
# we treat it as possible but not fully strong.
POSSIBLE_BUILDING_DISTANCE_M = 80.0


# ------------------------------------------------------------
# Optional utility: haversine distance
# ------------------------------------------------------------
# Not strictly needed right now because your input file already
# contains distance_m from the previous matching step.
# Keeping this here in case you want to reuse it later.
def haversine_m(lat1, lon1, lat2, lon2):
    """
    Compute distance in meters between two latitude/longitude points.
    """
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return float("inf")

    earth_radius_m = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_m * c


# ------------------------------------------------------------
# Building consistency classification
# ------------------------------------------------------------
def classify_building_relation(distance_m):
    """
    Use matched address distance as a simple proxy for building consistency.

    Interpretation:
    - <= 30m  -> likely same building / same parcel area
    - <= 80m  -> possibly still related, but less certain
    - > 80m   -> likely different building / weak relation
    """
    if pd.isna(distance_m):
        return "building_unknown"

    if distance_m <= SAME_BUILDING_DISTANCE_M:
        return "building_consistent"
    elif distance_m <= POSSIBLE_BUILDING_DISTANCE_M:
        return "building_possible"
    else:
        return "building_conflict"


# ------------------------------------------------------------
# Final label logic
# ------------------------------------------------------------
def assign_final_label(row):
    """
    Combine:
    - address match status
    - building consistency
    - shared-address clustering

    Goal:
    produce a more meaningful final confidence label.
    """

    status = row.get("status", "")
    building_status = row.get("building_status", "")
    cluster_size = row.get("address_cluster_size", 0)

    # Best case:
    # strong POI->address match and spatially very close
    if status == "matched_high" and building_status == "building_consistent":
        return "high_confidence_strong"

    # Good case:
    # matched_high or matched_medium, and not clearly conflicting
    if status in ["matched_high", "matched_medium"] and building_status != "building_conflict":
        return "medium_confidence_valid"

    # Shared commercial address case:
    # many POIs mapping to the same base address often indicates
    # malls, plazas, apartment complexes, outlet centers, etc.
    if cluster_size >= 5 and status in ["matched_medium", "uncertain"]:
        return "clustered_commercial_area"

    # Weak but still found something nearby
    if status == "uncertain":
        return "needs_review"

    # Everything else is low confidence
    return "low_confidence"


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def main():
    # Load results from previous POI -> address matching step
    df = pd.read_csv(INPUT_FILE)

    # --------------------------------------------------------
    # Step 1: classify building consistency using distance_m
    # --------------------------------------------------------
    df["building_status"] = df["distance_m"].apply(classify_building_relation)

    # --------------------------------------------------------
    # Step 2: detect shared-address clusters
    # --------------------------------------------------------
    # If many POIs map to the same address, that often indicates
    # a real shared address like a shopping center or mall.
    address_counts = df["best_address_text"].value_counts(dropna=False)
    df["address_cluster_size"] = df["best_address_text"].map(address_counts)

    # Fill any missing values just in case
    df["address_cluster_size"] = df["address_cluster_size"].fillna(0).astype(int)

    # --------------------------------------------------------
    # Step 3: assign final label
    # --------------------------------------------------------
    df["final_label"] = df.apply(assign_final_label, axis=1)

    # --------------------------------------------------------
    # Step 4: save full validated file
    # --------------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False)

    # --------------------------------------------------------
    # Step 5: also save split files for easier review
    # --------------------------------------------------------
    df[df["final_label"] == "high_confidence_strong"].to_csv(
        r"..\outputs\poi_address_high_confidence_strong.csv",
        index=False,
    )

    df[df["final_label"] == "medium_confidence_valid"].to_csv(
        r"..\outputs\poi_address_medium_confidence_valid.csv",
        index=False,
    )

    df[df["final_label"] == "clustered_commercial_area"].to_csv(
        r"..\outputs\poi_address_clustered_commercial_area.csv",
        index=False,
    )

    df[df["final_label"] == "needs_review"].to_csv(
        r"..\outputs\poi_address_needs_review.csv",
        index=False,
    )

    df[df["final_label"] == "low_confidence"].to_csv(
        r"..\outputs\poi_address_low_confidence.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Step 6: print summary
    # --------------------------------------------------------
    print(f"Saved building-validated results to: {OUTPUT_FILE}")
    print("\nFinal label distribution:")
    print(df["final_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()