import os
import geopandas as gpd
import pandas as pd

# Create folders
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/analysis", exist_ok=True)

# Load POIs
places = gpd.read_parquet("data/raw/places.parquet")
places = places.rename(columns={"id": "poi_id"})

print("Total POIs:", len(places))

# ----------------------------
# Step 1: Inspect categories
# ----------------------------
print("\nTop basic_category values:")
print(places["basic_category"].value_counts(dropna=False).head(50))

# ----------------------------
# Step 2: Define rules using REAL categories from your data
# ----------------------------
building_based = {
    "real_estate_service",
    "restaurant",
    "fashion_and_apparel_store",
    "financial_service",
    "professional_service",
    "personal_or_beauty_service",
    "social_or_community_service",
    "home_service",
    "college_university",
    "behavioral_or_mental_health_clinic",
    "complementary_and_alternative_medicine",
    "casual_eatery",
    "technical_service",
    "attorney_or_law_firm",
    "sporting_goods_store",
    "wellness_service",
    "bank_or_credit_union",
    "design_service",
    "corporate_or_business_office",
    "outpatient_care_facility",
    "bar",
    "food_and_beverage_store",
    "dental_clinic",
    "coffee_shop",
    "hardware_home_and_garden_store",
    "media_service",
    "specialized_health_care",
    "flowers_and_gifts_store",
}

site_based = {
    "park",
    "gas_station",
    "parking",
    "playground",
    "bus_stop",
    "charging_station",
    "sports_field",
    "trailhead",
    "cemetery",
}

building_based.update({"hotel", "cafe", "gym", "library", "hospital", "government_office", "art_gallery", "movie_theater", "museum", "event_venue", "fitness_studio", "convenience_store", "fast_food_restaurant"})
site_based.update({"recreational_trail_or_path"})

# ----------------------------
# Step 3: Classification logic
# ----------------------------
def classify_poi(category):
    if pd.isna(category):
        return "uncertain"

    category = str(category).lower().strip()

    if category in building_based:
        return "building_based"

    if category in site_based:
        return "site_based"

    return "uncertain"

places["eligibility"] = places["basic_category"].apply(classify_poi)

# ----------------------------
# Step 4: Summary stats
# ----------------------------
print("\nEligibility counts:")
print(places["eligibility"].value_counts())

# Show top uncertain categories so we can iteratively improve
uncertain = places[places["eligibility"] == "uncertain"].copy()
print("\nTop uncertain categories:")
print(uncertain["basic_category"].value_counts(dropna=False).head(50))

# ----------------------------
# Step 5: Save outputs
# ----------------------------
places.to_parquet("data/processed/poi_eligibility.parquet")

summary = places[["poi_id", "basic_category", "eligibility"]].copy()
summary.to_csv("data/analysis/poi_eligibility_summary.csv", index=False)

building_df = places[places["eligibility"] == "building_based"]
site_df = places[places["eligibility"] == "site_based"]
uncertain_df = places[places["eligibility"] == "uncertain"]

building_df.to_parquet("data/processed/poi_building_based.parquet")
site_df.to_parquet("data/processed/poi_site_based.parquet")
uncertain_df.to_parquet("data/processed/poi_uncertain.parquet")

print("\nSaved files:")
print("- data/processed/poi_eligibility.parquet")
print("- data/processed/poi_building_based.parquet")
print("- data/processed/poi_site_based.parquet")
print("- data/processed/poi_uncertain.parquet")
print("- data/analysis/poi_eligibility_summary.csv")