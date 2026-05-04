import pandas as pd
from shapely import wkb
from shapely.geometry import Point

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
INPUT_FILE = r"..\outputs\poi_address_building_validated.csv"
BUILDING_FILE = r"..\data\overture_buildings.parquet"
OUTPUT_FILE = r"..\outputs\poi_address_building_real_validated.csv"


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
def load_data():
    df = pd.read_csv(INPUT_FILE)
    buildings = pd.read_parquet(BUILDING_FILE)

    print("Loaded POI-address results:", len(df))
    print("Loaded buildings:", len(buildings))

    # Decode building geometry
    buildings["geometry_obj"] = buildings["geometry"].apply(
        lambda g: wkb.loads(g) if isinstance(g, (bytes, bytearray)) else g
    )

    return df, buildings


# ------------------------------------------------------------
# Check if point lies inside any building
# ------------------------------------------------------------
def find_building_for_point(lat, lon, buildings):
    if pd.isna(lat) or pd.isna(lon):
        return None

    point = Point(lon, lat)

    for _, b in buildings.iterrows():
        geom = b["geometry_obj"]
        if geom is not None and geom.contains(point):
            return b.get("id")

    return None


# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------
def main():
    df, buildings = load_data()

    poi_buildings = []
    address_buildings = []

    # --------------------------------------------------------
    # Step 1: POI → building
    # --------------------------------------------------------
    print("\nAssigning buildings to POIs...")

    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing row {i}")

        poi_lat = row.get("entrance_lat")
        poi_lon = row.get("entrance_lon")

        building_id = find_building_for_point(poi_lat, poi_lon, buildings)
        poi_buildings.append(building_id)

    df["poi_building_geom_id"] = poi_buildings

    # --------------------------------------------------------
    # Step 2: Address → building
    # --------------------------------------------------------
    print("\nAssigning buildings to matched addresses...")

    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing row {i}")

        lat = row.get("best_address_lat")
        lon = row.get("best_address_lon")

        building_id = find_building_for_point(lat, lon, buildings)
        address_buildings.append(building_id)

    # ✅ IMPORTANT FIX
    df["address_building_geom_id"] = address_buildings

    # --------------------------------------------------------
    # Step 3: Compare buildings
    # --------------------------------------------------------
    def compare_buildings(row):
        poi_b = row.get("poi_building_geom_id")
        addr_b = row.get("address_building_geom_id")

        if pd.isna(poi_b) or pd.isna(addr_b):
            return "building_unknown"

        if poi_b == addr_b:
            return "same_building"
        else:
            return "different_building"

    df["real_building_relation"] = df.apply(compare_buildings, axis=1)

    # --------------------------------------------------------
    # Step 4: Upgrade label
    # --------------------------------------------------------
    def upgraded_label(row):
        if row["real_building_relation"] == "same_building":
            if row["final_label"] in ["high_confidence_strong", "medium_confidence_valid"]:
                return "validated_strong"
            else:
                return "validated_spatial_match"

        if row["real_building_relation"] == "different_building":
            return "building_conflict_real"

        return row["final_label"]

    df["final_label_v2"] = df.apply(upgraded_label, axis=1)

    # --------------------------------------------------------
    # Step 5: Save results
    # --------------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False)

    # 🔥 NEW: save conflicts separately for debugging
    df[df["final_label_v2"] == "building_conflict_real"].to_csv(
        r"..\outputs\real_building_conflicts_for_review.csv",
        index=False,
    )

    print("\nSaved upgraded building validation results")
    print(df["final_label_v2"].value_counts())


if __name__ == "__main__":
    main()