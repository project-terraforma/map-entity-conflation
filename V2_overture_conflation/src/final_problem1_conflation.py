import pandas as pd

INPUT_FILE = "../outputs/poi_address_building_real_validated.csv"
OUTPUT_FILE = "../outputs/final_problem1_conflation.csv"

df = pd.read_csv(INPUT_FILE)

# ---------- Helper functions ----------

def is_missing(value):
    return pd.isna(value) or str(value).strip() == ""

def classify_poi(row):
    name = str(row.get("poi_name", row.get("name", ""))).lower()
    types = str(row.get("types", "")).lower()

    non_building_keywords = [
        "lake", "peak", "pass", "reservoir", "trail", "mountain",
        "park", "river", "creek", "campground", "viewpoint"
    ]

    text = name + " " + types

    if any(word in text for word in non_building_keywords):
        return "non_building_poi"

    return "building_expected_poi"

def final_label(row):
    poi_class = row["poi_class"]

    poi_address = row.get("poi_address_input", row.get("address", ""))
    best_address_id = row.get("best_address_id", "")
    final_label_v2 = str(row.get("final_label_v2", "")).lower()
    confidence = row.get("confidence", 0)

    if poi_class == "non_building_poi":
        return "non_building_poi"

    if "high_confidence_strong" in final_label_v2:
        return "same_building_confirmed"

    if "medium_confidence_valid" in final_label_v2:
        return "building_validated_candidate"

    if is_missing(poi_address) and not is_missing(best_address_id):
        return "spatial_address_candidate"

    if is_missing(poi_address) and is_missing(best_address_id):
        return "missing_poi_address_no_candidate"

    if confidence >= 0.7:
        return "text_and_distance_match"

    return "needs_review"

def match_method(row):
    poi_address = row.get("poi_address_input", row.get("address", ""))
    best_address_id = row.get("best_address_id", "")

    if row["final_label"] == "non_building_poi":
        return "not_applicable"

    if is_missing(poi_address) and not is_missing(best_address_id):
        return "spatial_only"

    if not is_missing(poi_address) and not is_missing(best_address_id):
        return "text_plus_distance"

    return "no_match"

def review_reason(row):
    label = row["final_label"]

    if label == "non_building_poi":
        return "POI type/name suggests this entity does not require building or address conflation."
    if label == "same_building_confirmed":
        return "POI and matched address are located in the same building polygon."
    if label == "building_validated_candidate":
        return "POI/address match is supported by building validation but not strong enough for highest confidence."
    if label == "spatial_address_candidate":
        return "POI has no address text, so match is based on nearest address/building spatial evidence."
    if label == "missing_poi_address_no_candidate":
        return "POI has no address text and no nearby address candidate was found."
    if label == "text_and_distance_match":
        return "POI address text and distance both support the match."
    return "Match requires manual review due to weak or conflicting evidence."


def extract_street_from_address(address):
    if is_missing(address):
        return None

    address = str(address).strip()

    # Example: "11578 W ARIZONA AVE" -> "W ARIZONA AVE"
    parts = address.split()

    if len(parts) <= 1:
        return None

    # Remove house number if first part is numeric
    if parts[0].isdigit():
        return " ".join(parts[1:])

    return address

# ---------- Add final labels ----------

df["poi_class"] = df.apply(classify_poi, axis=1)
df["final_label"] = df.apply(final_label, axis=1)
df["match_method"] = df.apply(match_method, axis=1)
df["review_reason"] = df.apply(review_reason, axis=1)

df["matched_street"] = df["best_address_text"].apply(extract_street_from_address)

def street_connection_status(row):
    if row["final_label"] == "non_building_poi":
        return "not_applicable"

    if is_missing(row.get("matched_street", "")):
        return "no_street_candidate"

    return "street_extracted_from_matched_address"

df["street_connection_status"] = df.apply(street_connection_status, axis=1)
# ---------- Build clean final output ----------

column_map = {
    "poi_id": "poi_id",
    "poi_name": "poi_name",
    "poi_address_input": "poi_address_input",
    "best_address_id": "matched_address_id",
    "best_address_text": "matched_address_text",
    "distance_m": "distance_m",
    "confidence": "confidence",
    "status": "address_match_status",
    "building_status": "building_status",
    "poi_building_geom_id": "matched_building_id",
    "real_building_relation": "real_building_relation",
    "final_label_v2": "building_validation_label",
    "matched_street": "matched_street",
    "street_connection_status": "street_connection_status",
}

available_cols = [col for col in column_map.keys() if col in df.columns]

final_df = df[available_cols + ["poi_class", "match_method", "final_label", "review_reason"]].copy()
final_df = final_df.rename(columns=column_map)

final_df.to_csv(OUTPUT_FILE, index=False)

print("Saved final Problem 1 output to:", OUTPUT_FILE)
print()
print(final_df["final_label"].value_counts())
print()

preview_cols = [
    "poi_name",
    "matched_address_text",
    "matched_street",
    "street_connection_status",
    "match_method",
    "final_label"
]

existing_preview_cols = [col for col in preview_cols if col in final_df.columns]

print(final_df[existing_preview_cols].head(20))
