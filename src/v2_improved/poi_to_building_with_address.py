import os
import geopandas as gpd
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
poi_matches = gpd.read_parquet("data/processed/poi_building_scored.parquet")
address_matches = gpd.read_parquet("data/processed/address_building_matches.parquet")

# Normalize building column
building_col = None
for col in ["building_id", "building_id_right"]:
    if col in poi_matches.columns:
        building_col = col
        break

if building_col is None:
    raise ValueError("No building id column in POI matches")

poi_matches = poi_matches.rename(columns={building_col: "building_id"})

print("Loaded POI matches:", len(poi_matches))
print("Loaded address matches:", len(address_matches))

# ----------------------------
# Step 1: Build building → address count map
# ----------------------------
building_address_count = (
    address_matches.groupby("building_id")
    .size()
    .reset_index(name="num_addresses")
)

# Merge into POI matches
poi = poi_matches.merge(building_address_count, on="building_id", how="left")

poi["num_addresses"] = poi["num_addresses"].fillna(0)

# ----------------------------
# Step 2: Address signal from POI
# ----------------------------
def has_address(addr):
    return addr is not None and str(addr) != "nan"

poi["has_poi_address"] = poi["addresses"].apply(has_address)

# ----------------------------
# Step 3: Compute final score
# ----------------------------
def compute_final_score(row):
    base = row["score"]

    # boost if building has addresses
    building_bonus = 0.1 if row["num_addresses"] > 0 else 0

    # boost if POI has address info
    poi_bonus = 0.1 if row["has_poi_address"] else 0

    return base + building_bonus + poi_bonus

poi["final_score"] = poi.apply(compute_final_score, axis=1)

# ----------------------------
# Step 4: Normalize score (cap at 1)
# ----------------------------
poi["final_score"] = poi["final_score"].clip(upper=1.0)

# ----------------------------
# Stats
# ----------------------------
print("\nFinal score summary:")
print(poi["final_score"].describe())

print("\nAddress signal:")
print("POIs with address:", poi["has_poi_address"].sum())

print("\nBuildings with addresses:")
print((poi["num_addresses"] > 0).sum())

# ----------------------------
# Save
# ----------------------------
poi.to_parquet("data/processed/poi_building_final.parquet")

print("\nSaved: data/processed/poi_building_final.parquet")