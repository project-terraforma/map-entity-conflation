import os
import geopandas as gpd
import pandas as pd
import re

os.makedirs("data/processed", exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
poi_matches = gpd.read_parquet("data/processed/poi_building_scored.parquet")
address_matches = gpd.read_parquet("data/processed/address_building_matches.parquet")

# Normalize building id column
building_col = None
for col in ["building_id", "building_id_right"]:
    if col in poi_matches.columns:
        building_col = col
        break

poi_matches = poi_matches.rename(columns={building_col: "building_id"})

# ----------------------------
# Step 1: Clean building addresses
# ----------------------------
addr = address_matches.copy()

# Extract useful fields (if they exist)
cols = [c for c in ["number", "street", "postcode"] if c in addr.columns]
addr = addr[["building_id"] + cols]

# Normalize text
def clean(x):
    if pd.isna(x):
        return None
    return str(x).lower().strip()

for c in cols:
    addr[c] = addr[c].apply(clean)

# Group addresses per building
building_addr = addr.groupby("building_id").agg(list).reset_index()

# ----------------------------
# Step 2: Extract POI address
# ----------------------------
def extract_poi_address(addr_array):
    if addr_array is None or len(addr_array) == 0:
        return None, None, None

    a = addr_array[0]

    free = a.get("freeform", None)
    postcode = a.get("postcode", None)

    if free is None:
        return None, None, postcode

    free = free.lower()

    # extract number
    num_match = re.match(r"\d+", free)
    number = num_match.group() if num_match else None

    # extract street (remove number)
    street = re.sub(r"^\d+\s*", "", free)

    return number, street, postcode

poi_matches[["poi_number", "poi_street", "poi_postcode"]] = (
    poi_matches["addresses"]
    .apply(lambda x: pd.Series(extract_poi_address(x)))
)

# ----------------------------
# Step 3: Merge building address
# ----------------------------
poi = poi_matches.merge(building_addr, on="building_id", how="left")

# ----------------------------
# Step 4: Address matching score
# ----------------------------
def address_score(row):
    score = 0

    building_numbers = row.get("number", [])
    building_streets = row.get("street", [])
    building_postcodes = row.get("postcode", [])

    # convert NaN → empty list
    if not isinstance(building_numbers, list):
        building_numbers = []
    if not isinstance(building_streets, list):
        building_streets = []
    if not isinstance(building_postcodes, list):
        building_postcodes = []

    # number match (strong)
    if row["poi_number"]:
        if row["poi_number"] in building_numbers:
            score += 0.5

    # street match (medium)
    if row["poi_street"]:
        for s in building_streets:
            if row["poi_street"] in s:
                score += 0.3
                break

    # postcode match (weak)
    if row["poi_postcode"]:
        if row["poi_postcode"] in building_postcodes:
            score += 0.2

    return score

poi["address_score"] = poi.apply(address_score, axis=1)

# ----------------------------
# Step 5: Final score
# ----------------------------
poi["final_score"] = poi["score"] + poi["address_score"]
poi["final_score"] = poi["final_score"].clip(upper=1.5)

# ----------------------------
# Stats
# ----------------------------
print("\nAddress score summary:")
print(poi["address_score"].describe())

print("\nFinal score summary:")
print(poi["final_score"].describe())

print("\nHigh address matches (>0.5):")
print((poi["address_score"] > 0.5).sum())

# ----------------------------
# Save
# ----------------------------
poi.to_parquet("data/processed/poi_building_address_matched.parquet")

print("\nSaved: data/processed/poi_building_address_matched.parquet")