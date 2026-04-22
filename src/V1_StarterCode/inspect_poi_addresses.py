import geopandas as gpd
import pandas as pd

pois = gpd.read_parquet("data/processed/poi_building_based.parquet")

print("Columns:")
print(pois.columns.tolist())

print("\nSample POI addresses:")
for i, val in enumerate(pois["addresses"].head(10)):
    print(f"\n--- Address sample {i+1} ---")
    print(val)
    print("Type:", type(val))