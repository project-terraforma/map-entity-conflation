import os
import geopandas as gpd
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
pois = gpd.read_parquet("data/processed/poi_building_based.parquet")
buildings = gpd.read_parquet("data/raw/buildings.parquet")

pois = pois.rename(columns={"id": "poi_id"})
buildings = buildings.rename(columns={"id": "building_id"})

# Project to metric CRS
pois = pois.to_crs(epsg=26913)
buildings = buildings.to_crs(epsg=26913)

print("Building-based POIs:", len(pois))

# ----------------------------
# Step 1: within matches
# ----------------------------
within = gpd.sjoin(
    pois,
    buildings,
    predicate="within",
    how="left"
)

within["match_type"] = "within"
within["distance"] = 0

# Separate unmatched
matched_within = within[within["index_right"].notna()].copy()
unmatched = within[within["index_right"].isna()].copy()

unmatched = unmatched.drop(columns=["index_right"])

# ----------------------------
# Step 2: nearest candidates
# ----------------------------
nearest = gpd.sjoin_nearest(
    unmatched,
    buildings,
    how="left",
    distance_col="distance"
)

nearest["match_type"] = "nearest"

# Keep only reasonable candidates (IMPORTANT)
nearest = nearest[nearest["distance"] <= 30]

# Keep best nearest per POI
nearest = nearest.sort_values("distance").drop_duplicates(subset="poi_id")

# ----------------------------
# Step 3: Combine
# ----------------------------
combined = pd.concat([matched_within, nearest], ignore_index=True)
combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=pois.crs)

# ----------------------------
# Step 4: Scoring
# ----------------------------
def compute_score(row):
    if row["match_type"] == "within":
        return 1.0  # best possible

    d = row["distance"]

    # convert distance to score (closer = higher)
    return max(0, 1 - (d / 30))

combined["score"] = combined.apply(compute_score, axis=1)

# ----------------------------
# Step 5: Final selection
# ----------------------------
final = combined.sort_values("score", ascending=False).drop_duplicates(subset="poi_id")

# ----------------------------
# Stats
# ----------------------------
print("\nFinal stats:")
print("Total matched:", len(final))
print("Unique POIs:", final["poi_id"].nunique())

print("\nMatch type counts:")
print(final["match_type"].value_counts())

print("\nScore summary:")
print(final["score"].describe())

# ----------------------------
# Save
# ----------------------------
final.to_parquet("data/processed/poi_building_scored.parquet")

print("\nSaved: data/processed/poi_building_scored.parquet")