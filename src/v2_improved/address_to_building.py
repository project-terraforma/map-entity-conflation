import os
import geopandas as gpd
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
addresses = gpd.read_parquet("data/raw/addresses.parquet")
buildings = gpd.read_parquet("data/raw/buildings.parquet")

addresses = addresses.rename(columns={"id": "address_id"})
buildings = buildings.rename(columns={"id": "building_id"})

# Project to metric CRS
addresses = addresses.to_crs(epsg=26913)
buildings = buildings.to_crs(epsg=26913)

print("Total addresses:", len(addresses))

# ----------------------------
# Step 1: within match
# ----------------------------
within = gpd.sjoin(
    addresses,
    buildings,
    predicate="within",
    how="left"
)

within["match_type"] = "within"
within["distance"] = 0

matched_within = within[within["index_right"].notna()].copy()
unmatched = within[within["index_right"].isna()].copy()

unmatched = unmatched.drop(columns=["index_right"])

# ----------------------------
# Step 2: nearest fallback
# ----------------------------
nearest = gpd.sjoin_nearest(
    unmatched,
    buildings,
    how="left",
    distance_col="distance"
)

nearest["match_type"] = "nearest"

# Keep only close matches (addresses should be VERY close)
nearest = nearest[nearest["distance"] <= 20]

# Keep best per address
nearest = nearest.sort_values("distance").drop_duplicates(subset="address_id")

# ----------------------------
# Combine
# ----------------------------
combined = pd.concat([matched_within, nearest], ignore_index=True)
combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=addresses.crs)


combined = combined.sort_values("distance").drop_duplicates(subset="address_id")
# ----------------------------
# Stats
# ----------------------------
print("\nFinal stats:")
print("Total matched:", len(combined))
print("Unique addresses:", combined["address_id"].nunique())

print("\nMatch types:")
print(combined["match_type"].value_counts())

print("\nDistance summary:")
if "distance" in combined.columns:
    print(combined["distance"].describe())


# ----------------------------
# Save
# ----------------------------
combined.to_parquet("data/processed/address_building_matches.parquet")

print("\nSaved: data/processed/address_building_matches.parquet")