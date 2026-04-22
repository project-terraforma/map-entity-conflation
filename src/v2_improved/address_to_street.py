import os
import geopandas as gpd
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
addresses = gpd.read_parquet("data/raw/addresses.parquet")
streets = gpd.read_parquet("data/raw/streets.parquet")  # assuming you have this

addresses = addresses.rename(columns={"id": "address_id"})
streets = streets.rename(columns={"id": "street_id"})

# Project
addresses = addresses.to_crs(epsg=26913)
streets = streets.to_crs(epsg=26913)

print("Total addresses:", len(addresses))

# ----------------------------
# Nearest street
# ----------------------------
addr_street = gpd.sjoin_nearest(
    addresses,
    streets,
    how="left",
    distance_col="distance_to_street"
)

# Keep reasonable matches
addr_street = addr_street[addr_street["distance_to_street"] <= 30]

# Deduplicate
addr_street = addr_street.sort_values("distance_to_street").drop_duplicates(subset="address_id")

# ----------------------------
# Stats
# ----------------------------
print("\nMatches:", len(addr_street))
print("Unique addresses:", addr_street["address_id"].nunique())

print("\nDistance summary:")
print(addr_street["distance_to_street"].describe())

# ----------------------------
# Save
# ----------------------------
addr_street.to_parquet("data/processed/address_street_matches.parquet")

print("\nSaved: data/processed/address_street_matches.parquet")