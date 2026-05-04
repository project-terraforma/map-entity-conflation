# This file loads the raw preprocessed POI tile JSON files and turns them into
# one flat table for analysis.
#
# In simple words, it:
# 1. opens every JSON tile file
# 2. reads the "waypoints" list inside each file
# 3. pulls out the POI fields we care about
# 4. converts each POI into one row
# 5. combines all rows into one pandas DataFrame
# 6. saves the final result as a CSV
#
# This matters because the original data is spread across many nested JSON files,
# which is hard to analyze directly. This file makes the data much easier to use.

import json
from pathlib import Path
import pandas as pd


# Helper function:
# Some useful metadata is stored inside waypoint["c"].
# This safely gets a value from that nested dictionary.
def safe_get_context(waypoint, key):
    context = waypoint.get("c", {})
    return context.get(key)


# This function loads one tile JSON file and extracts all waypoint rows from it.
def load_waypoints_from_tile(tile_path: Path):
    # Open the tile file and read the JSON content into Python
    with open(tile_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get the list of POIs/waypoints from this tile
    waypoints = data.get("waypoints", [])
    rows = []

    # Go through each waypoint and flatten its fields into one row
    for wp in waypoints:
        row = {
            # Basic source information
            "tile_file": tile_path.name,
            "id": wp.get("id"),
            "name": wp.get("name"),

            # Original POI coordinates
            "latitude": wp.get("latitude"),
            "longitude": wp.get("longitude"),

            # Corrected entrance coordinates
            "entrance_lat": wp.get("entrance_lat"),
            "entrance_lon": wp.get("entrance_lon"),

            # Address / category / building link
            "address": wp.get("address"),
            "types": str(wp.get("types")),
            "overture_building_id": wp.get("overture_building_id"),

            # Context fields from waypoint["c"]
            "entrance_type": safe_get_context(wp, "et"),
            "entrance_street": safe_get_context(wp, "es"),
            "facade_bearing": safe_get_context(wp, "fb"),
            "facade_cardinal": safe_get_context(wp, "fc"),
            "venue_type": safe_get_context(wp, "vt"),
            "osm_confirmed": safe_get_context(wp, "osm"),
            "osm_match_tier": safe_get_context(wp, "ot"),
            "co_tenants": safe_get_context(wp, "ct"),
            "building_tenants": safe_get_context(wp, "bt"),
            "address_street_mismatch": safe_get_context(wp, "asm"),
        }

        # Add this flattened POI row to the list
        rows.append(row)

    return rows


# This function loads every tile file in the z14 folder
# and combines all rows into one DataFrame.
def load_all_tiles(z14_dir: Path):
    all_rows = []

    # Find all JSON files in the tile directory
    json_files = sorted(z14_dir.glob("*.json"))
    print(f"Found {len(json_files)} tile files")

    # Load one tile at a time
    for tile_file in json_files:
        try:
            rows = load_waypoints_from_tile(tile_file)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Failed on {tile_file.name}: {e}")

    # Convert the list of dictionaries into a pandas table
    df = pd.DataFrame(all_rows)
    return df


# Main script execution
if __name__ == "__main__":
    # BASE_DIR points to the root of the project folder
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Input folder containing the tile JSON files
    z14_dir = BASE_DIR / "overture-poi-preprocessing" / "z14"

    # Output CSV path
    output_csv = BASE_DIR / "outputs" / "poi_waypoints_flat.csv"

    # Load and combine all tile files
    df = load_all_tiles(z14_dir)

    # Preview the first few rows and show shape
    print(df.head())
    print(df.shape)

    # Make sure the output folder exists, then save the CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved to {output_csv}")