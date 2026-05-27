"""Orchestrate the Problem 2 facade matching pipeline."""
import pandas as pd

from data_loader import load_layers
from facade_matcher import run_matching
from poi_building_linker import link_pois_to_buildings
from summary_writer import write_debug_sample, write_manual_sample, write_matches, write_summary


def run():
    """Run inspection, linking, facade extraction/matching, and output writing."""
    print("Starting Problem 2: POI <-> building facade matching")
    layers, inspection = load_layers()

    try:
        poi_gdf, building_gdf, entrance_gdf, streets_gdf, notes = link_pois_to_buildings(layers)
    except Exception as exc:
        notes = [f"linking failed: {exc}"]
        print(notes[0])
        write_summary(inspection, notes, pd.DataFrame())
        return

    if poi_gdf is None or building_gdf is None:
        notes.append("Insufficient spatial data to run matching. Add places and building footprint files, then rerun.")
        write_summary(inspection, notes, pd.DataFrame())
        print("Problem 2 pipeline stopped after inspection because required spatial layers were unavailable.")
        return

    print(f"Running facade matching on {len(poi_gdf)} POIs and {len(building_gdf)} buildings")
    records = run_matching(poi_gdf, building_gdf, entrances_gdf=entrance_gdf, streets_gdf=streets_gdf)
    matches_df = pd.DataFrame(records)
    metric_crs = poi_gdf.crs

    public_df = write_matches(matches_df, metric_crs)
    write_summary(inspection, notes, public_df)
    write_manual_sample(public_df)
    write_debug_sample(matches_df, metric_crs)
    print("Problem 2 pipeline finished.")


if __name__ == "__main__":
    run()
