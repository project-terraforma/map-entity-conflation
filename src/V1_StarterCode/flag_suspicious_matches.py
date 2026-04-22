import os
import geopandas as gpd
import pandas as pd

os.makedirs("data/analysis", exist_ok=True)

matches = gpd.read_parquet("data/processed/poi_building_address_matched.parquet")

def label_match(row):
    geom_score = row["score"]
    addr_score = row["address_score"]
    match_type = row["match_type"]

    # strongest cases
    if match_type == "within" and addr_score >= 0.5:
        return "strong_match"

    # good within matches even without address support
    if match_type == "within" and addr_score < 0.5:
        return "likely_match"

    # nearest but strongly supported by address
    if match_type == "nearest" and addr_score >= 0.5 and geom_score >= 0.6:
        return "likely_match"

    # nearest with poor address support
    if match_type == "nearest" and addr_score == 0 and geom_score < 0.6:
        return "likely_wrong"

    # everything else
    return "needs_review"

matches["match_label"] = matches.apply(label_match, axis=1)

print("Match label counts:")
print(matches["match_label"].value_counts())

# save full labeled file
matches.to_parquet("data/processed/poi_building_labeled.parquet")

# save suspicious cases
suspicious = matches[matches["match_label"].isin(["likely_wrong", "needs_review"])].copy()

cols_to_save = [
    "poi_id",
    "match_type",
    "score",
    "address_score",
    "final_score",
    "match_label",
]

for c in ["names", "basic_category", "building_id", "building_id_right"]:
    if c in suspicious.columns and c not in cols_to_save:
        cols_to_save.append(c)

suspicious[cols_to_save].to_csv("data/analysis/suspicious_matches.csv", index=False)

print("\nSaved:")
print("- data/processed/poi_building_labeled.parquet")
print("- data/analysis/suspicious_matches.csv")