"""Write Problem 2 outputs, summaries, review samples, and debug GeoJSON."""
from pathlib import Path

import pandas as pd

from config import (
    DEBUG_SAMPLE_GEOJSON,
    DEBUG_SAMPLE_SIZE,
    FACADE_MATCHES_CSV,
    FACADE_MATCHES_GEOJSON,
    MANUAL_SAMPLE_SIZE,
    REVIEW_SAMPLE,
    SUMMARY_CSV,
)
from geometry_utils import safe_wkt


INTERNAL_GEOMETRY_COLUMNS = [
    "_selected_facade_geometry",
    "_candidate_facade_geometries",
    "_building_geometry",
    "_poi_geometry",
]


def ensure_output_dir():
    Path(FACADE_MATCHES_CSV.parent).mkdir(parents=True, exist_ok=True)


def _transform_match_geometries(df, source_crs):
    """Convert internal projected geometries to WGS84 output fields."""
    if df.empty or "_selected_facade_geometry" not in df.columns:
        return df, None
    try:
        import geopandas as gpd

        result = df.copy()
        facade_mask = result["_selected_facade_geometry"].notna()
        facade_projected = gpd.GeoDataFrame(
            result.loc[facade_mask].copy(),
            geometry="_selected_facade_geometry",
            crs=source_crs,
        )
        midpoint_projected = facade_projected.geometry.interpolate(0.5, normalized=True)
        facade_gdf = facade_projected.to_crs("EPSG:4326")
        result.loc[facade_mask, "facade_wkt"] = facade_gdf.geometry.apply(safe_wkt).values
        midpoints = gpd.GeoSeries(midpoint_projected, crs=source_crs).to_crs("EPSG:4326")
        result.loc[facade_mask, "facade_midpoint_lon"] = midpoints.x.values
        result.loc[facade_mask, "facade_midpoint_lat"] = midpoints.y.values

        poi_mask = result["_poi_geometry"].notna()
        poi_gdf = gpd.GeoDataFrame(result.loc[poi_mask].copy(), geometry="_poi_geometry", crs=source_crs).to_crs("EPSG:4326")
        result.loc[poi_mask, "poi_lon"] = poi_gdf.geometry.x.values
        result.loc[poi_mask, "poi_lat"] = poi_gdf.geometry.y.values

        geojson_gdf = gpd.GeoDataFrame(facade_gdf.copy(), geometry=facade_gdf.geometry, crs="EPSG:4326")
        return result, geojson_gdf
    except Exception as exc:
        print(f"Could not transform output geometries to WGS84: {exc}")
        return df, None


def public_columns(df):
    """Remove private geometry helper columns before CSV output."""
    return [col for col in df.columns if col not in INTERNAL_GEOMETRY_COLUMNS]


def write_matches(df, source_crs):
    """Write CSV and selected-facade GeoJSON outputs."""
    ensure_output_dir()
    output_df, facade_gdf = _transform_match_geometries(df, source_crs)
    output_df[public_columns(output_df)].to_csv(FACADE_MATCHES_CSV, index=False)
    print(f"Wrote matches CSV to {FACADE_MATCHES_CSV}")
    if facade_gdf is not None and not facade_gdf.empty:
        facade_gdf[public_columns(facade_gdf)].to_file(FACADE_MATCHES_GEOJSON, driver="GeoJSON")
        print(f"Wrote facade GeoJSON to {FACADE_MATCHES_GEOJSON}")
    else:
        print("No selected facade geometries available for GeoJSON output.")
    return output_df[public_columns(output_df)]


def write_summary(inspection_records, notes, matches_df):
    """Write a compact summary CSV with layer inspection and match counts."""
    ensure_output_dir()
    rows = list(inspection_records)
    for note in notes:
        rows.append({"layer": "pipeline_note", "source_path": "", "rows": "", "columns": "", "status": note})
    if matches_df is not None and not matches_df.empty:
        counts = matches_df["confidence_label"].value_counts(dropna=False).reset_index()
        counts.columns = ["confidence_label", "count"]
        for _, row in counts.iterrows():
            rows.append(
                {
                    "layer": "match_count",
                    "source_path": "",
                    "rows": int(row["count"]),
                    "columns": "",
                    "status": row["confidence_label"],
                }
            )
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote summary to {SUMMARY_CSV}")


def write_manual_sample(df):
    """Write a deterministic sample for human review."""
    ensure_output_dir()
    public_df = df[public_columns(df)] if df is not None and not df.empty else pd.DataFrame()
    sample = public_df.sample(n=min(MANUAL_SAMPLE_SIZE, len(public_df)), random_state=42) if len(public_df) > 0 else public_df
    sample.to_csv(REVIEW_SAMPLE, index=False)
    print(f"Wrote manual review sample to {REVIEW_SAMPLE}")


def write_debug_sample(df, source_crs):
    """Write POI, building, and facade features for a small visual QA sample."""
    if df is None or df.empty or "_selected_facade_geometry" not in df.columns:
        return
    try:
        import geopandas as gpd

        rows = []
        for _, row in df.head(DEBUG_SAMPLE_SIZE).iterrows():
            for feature_type, geom_col in [
                ("poi", "_poi_geometry"),
                ("building", "_building_geometry"),
                ("selected_facade", "_selected_facade_geometry"),
            ]:
                geom = row.get(geom_col)
                if geom is None:
                    continue
                rows.append(
                    {
                        "poi_id": row.get("poi_id", ""),
                        "matched_building_id": row.get("matched_building_id", ""),
                        "feature_type": feature_type,
                        "confidence_label": row.get("confidence_label", ""),
                        "geometry": geom,
                    }
                )
            for facade_index, geom in enumerate(row.get("_candidate_facade_geometries") or []):
                rows.append(
                    {
                        "poi_id": row.get("poi_id", ""),
                        "matched_building_id": row.get("matched_building_id", ""),
                        "feature_type": "candidate_facade",
                        "candidate_facade_index": facade_index,
                        "confidence_label": row.get("confidence_label", ""),
                        "geometry": geom,
                    }
                )
        if not rows:
            return
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=source_crs).to_crs("EPSG:4326")
        gdf.to_file(DEBUG_SAMPLE_GEOJSON, driver="GeoJSON")
        print(f"Wrote debug sample GeoJSON to {DEBUG_SAMPLE_GEOJSON}")
    except Exception as exc:
        print(f"Could not write debug sample GeoJSON: {exc}")
