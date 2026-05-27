"""Create static visual examples for Problem 2 facade matching.

This script is visualization-only. It reconstructs candidate facade edges from
existing building geometry and writes PNG examples plus a caption index.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from shapely import wkt
from shapely.geometry import Point

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from config import BUILDINGS_INPUT_OPTIONS, OUTPUT_DIR, STREETS_INPUT_OPTIONS
from data_loader import find_input_file, load_table, parse_geometry
from facade_extractor import extract_facades_from_building
from geometry_utils import safe_wkt, to_metric_crs
from improved_facade_ranker import (
    candidate_record,
    new_label_for_row,
    old_facade_crowding,
)


MATCHES_CSV = OUTPUT_DIR / "problem2_facade_matches.csv"
VIS_DIR = OUTPUT_DIR / "visualizations"
CAPTIONS_CSV = VIS_DIR / "captions.csv"

MAX_PER_GROUP = 5
PLOT_BUFFER_M = 45.0


def note(message):
    print(f"NOTE: {message}")


def load_matches():
    if not MATCHES_CSV.exists():
        raise FileNotFoundError(f"Problem 2 matches not found: {MATCHES_CSV}")
    df = pd.read_csv(MATCHES_CSV, low_memory=False)
    print(f"Loaded matches: {MATCHES_CSV} ({len(df)} rows)")
    print(f"Match columns: {list(df.columns)}")
    return df


def load_buildings():
    if gpd is None:
        raise ImportError("geopandas is required for visualizations.")
    path = find_input_file(BUILDINGS_INPUT_OPTIONS, "buildings", required=False)
    if path is None:
        raise FileNotFoundError("Building geometry file not found.")
    df = load_table(path)
    print(f"Loaded buildings: {path} ({len(df)} rows)")
    print(f"Building columns: {list(df.columns)}")
    if "id" not in df.columns or "geometry" not in df.columns:
        raise ValueError("Building file must contain existing columns 'id' and 'geometry'.")
    out = df[["id", "geometry"]].copy()
    out["geometry"] = out["geometry"].apply(parse_geometry)
    out = out[out["geometry"].apply(lambda geom: geom is not None and not geom.is_empty)].copy()
    return gpd.GeoDataFrame(out.rename(columns={"id": "building_id"}), geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326")


def load_streets(metric_crs):
    path = find_input_file(STREETS_INPUT_OPTIONS, "streets", required=False)
    if path is None:
        note("No street/segment file found; street overlays will be omitted.")
        return None
    df = load_table(path)
    print(f"Loaded streets: {path} ({len(df)} rows)")
    print(f"Street columns: {list(df.columns)}")
    if "geometry" not in df.columns:
        note("Street file has no geometry column; street overlays will be omitted.")
        return None
    out = df[["geometry"]].copy()
    out["geometry"] = out["geometry"].apply(parse_geometry)
    out = out[out["geometry"].apply(lambda geom: geom is not None and not geom.is_empty and geom.geom_type in {"LineString", "MultiLineString"})].copy()
    if out.empty:
        note("No valid street line geometries found; street overlays will be omitted.")
        return None
    return gpd.GeoDataFrame(out, geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326").to_crs(metric_crs)


def parse_wkt(value):
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    try:
        return wkt.loads(str(value))
    except Exception:
        return None


def metric_point(row, metric_crs):
    lon = pd.to_numeric(pd.Series([row.get("poi_lon")]), errors="coerce").iloc[0]
    lat = pd.to_numeric(pd.Series([row.get("poi_lat")]), errors="coerce").iloc[0]
    if pd.isna(lon) or pd.isna(lat):
        return None
    point = gpd.GeoDataFrame([{"geometry": Point(lon, lat)}], geometry="geometry", crs="EPSG:4326").to_crs(metric_crs)
    return point.geometry.iloc[0]


def old_selected_geometry(row, metric_crs):
    geom = parse_wkt(row.get("facade_wkt"))
    if geom is None:
        return None
    return gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(metric_crs).iloc[0]


def nearest_streets(streets_gdf, geometry, limit=3):
    if streets_gdf is None or streets_gdf.empty or geometry is None:
        return None
    try:
        search = geometry.buffer(PLOT_BUFFER_M)
        indices = list(streets_gdf.sindex.query(search, predicate="intersects"))
        if not indices:
            nearest = streets_gdf.sindex.nearest(geometry, return_all=False)
            indices = list(nearest[1]) if len(nearest) >= 2 else []
        if not indices:
            return None
        subset = streets_gdf.iloc[indices].copy()
        subset["_distance"] = subset.geometry.distance(geometry)
        return subset.sort_values("_distance").head(limit)
    except Exception:
        return None


def rerank_one(row, building_geom, streets_gdf, crowding, metric_crs):
    point = metric_point(row, metric_crs)
    if point is None:
        return None
    poi_row = pd.Series({"geometry": point})
    building_id = str(row.get("matched_building_id", "") or "")
    facades = extract_facades_from_building(building_geom)
    if not facades:
        return None
    candidates = [candidate_record(poi_row, building_id, facade, streets_gdf, crowding) for facade in facades]
    ranked = sorted(candidates, key=lambda item: item["adjusted_rank_score_m"])
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    new_label, evidence = new_label_for_row(poi_row, building_geom, best, second)
    return {
        "new_facade_id": best["facade_id"],
        "new_facade_index": best["facade_index"],
        "new_confidence_label": new_label,
        "new_ranker_evidence": evidence,
        "new_rank_gap_m": round(second["adjusted_rank_score_m"] - best["adjusted_rank_score_m"], 3) if second else None,
        "new_selected_geometry": facades[best["facade_index"]]["geometry"],
    }


def endpoint_distance(row, metric_crs):
    point = metric_point(row, metric_crs)
    line = old_selected_geometry(row, metric_crs)
    if point is None or line is None:
        return None
    nearest = line.interpolate(line.project(point))
    coords = list(line.coords)
    return min(nearest.distance(Point(coords[0])), nearest.distance(Point(coords[-1])))


def pick_examples(matches, buildings_m, streets_m):
    building_ids = set(buildings_m["building_id"].astype(str))
    valid = matches[matches["matched_building_id"].fillna("").astype(str).isin(building_ids)].copy()

    strong = valid[valid["confidence_label"].eq("high_confidence_nearest_facade")].copy()
    strong = strong.sort_values("distance_to_facade_meters").head(MAX_PER_GROUP)

    ambiguous = valid[valid["confidence_label"].eq("needs_review_multiple_close_facades")].copy()
    ambiguous = ambiguous.sort_values("second_best_facade_distance_m").head(MAX_PER_GROUP)

    corner_rows = ambiguous.copy()
    corner_rows["_endpoint_distance_m"] = corner_rows.apply(lambda row: endpoint_distance(row, buildings_m.crs), axis=1)
    corner = corner_rows.dropna(subset=["_endpoint_distance_m"]).sort_values("_endpoint_distance_m").head(MAX_PER_GROUP)

    tenant_counts = valid["matched_building_id"].fillna("").astype(str).value_counts()
    multi = valid.copy()
    multi["_pois_in_building"] = multi["matched_building_id"].fillna("").astype(str).map(tenant_counts)
    multi = multi[multi["_pois_in_building"].gt(1)].sort_values("_pois_in_building", ascending=False).head(MAX_PER_GROUP)

    crowding = old_facade_crowding(matches)
    building_lookup = buildings_m.drop_duplicates("building_id").set_index("building_id")
    improved_records = []
    for _, row in ambiguous.iterrows():
        building_id = str(row.get("matched_building_id", "") or "")
        reranked = rerank_one(row, building_lookup.loc[building_id].geometry, streets_m, crowding, buildings_m.crs)
        if not reranked:
            continue
        if reranked["new_confidence_label"] != row.get("confidence_label") or reranked["new_facade_id"] != row.get("facade_id"):
            improved = row.copy()
            for key, value in reranked.items():
                improved[key] = value
            improved_records.append(improved)
        if len(improved_records) >= MAX_PER_GROUP:
            break
    improved = pd.DataFrame(improved_records)

    return {
        "strong": strong,
        "ambiguous": ambiguous,
        "improved_after_reranking": improved,
        "corner_ambiguity": corner,
        "multi_tenant": multi,
    }


def plot_example(row, group_name, index, buildings_m, streets_m):
    building_id = str(row.get("matched_building_id", "") or "")
    building_lookup = buildings_m.drop_duplicates("building_id").set_index("building_id")
    if building_id not in building_lookup.index:
        return None

    building_geom = building_lookup.loc[building_id].geometry
    point = metric_point(row, buildings_m.crs)
    if point is None:
        return None

    facades = extract_facades_from_building(building_geom)
    selected_old = old_selected_geometry(row, buildings_m.crs)
    selected_new = row.get("new_selected_geometry") if "new_selected_geometry" in row.index else None
    street_subset = nearest_streets(streets_m, building_geom)

    fig, ax = plt.subplots(figsize=(7, 7))
    gpd.GeoSeries([building_geom], crs=buildings_m.crs).plot(ax=ax, color="#ece7df", edgecolor="#303030", linewidth=1.2)

    if street_subset is not None and not street_subset.empty:
        street_subset.plot(ax=ax, color="#6f4aa8", linewidth=1.6, alpha=0.75)

    if facades:
        gpd.GeoSeries([facade["geometry"] for facade in facades], crs=buildings_m.crs).plot(ax=ax, color="#999999", linewidth=1.0, alpha=0.65)

    if selected_old is not None:
        gpd.GeoSeries([selected_old], crs=buildings_m.crs).plot(ax=ax, color="#d62728", linewidth=4.0)

    if selected_new is not None:
        gpd.GeoSeries([selected_new], crs=buildings_m.crs).plot(ax=ax, color="#2ca02c", linewidth=3.0, linestyle="--")

    ax.scatter([point.x], [point.y], color="#1f77b4", s=70, zorder=5, marker="o", edgecolor="white", linewidth=1.0)

    minx, miny, maxx, maxy = building_geom.buffer(PLOT_BUFFER_M).bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    title = f"{group_name.replace('_', ' ').title()} #{index}"
    subtitle = caption_for(row, group_name, len(facades))
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    output_path = VIS_DIR / f"{group_name}_{index:02d}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path, subtitle


def caption_for(row, group_name, facade_count):
    label = row.get("confidence_label", "")
    distance = row.get("distance_to_facade_meters", "")
    gap = ""
    if pd.notna(row.get("second_best_facade_distance_m")) and pd.notna(row.get("distance_to_facade_meters")):
        gap_value = pd.to_numeric(pd.Series([row.get("second_best_facade_distance_m")]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([row.get("distance_to_facade_meters")]), errors="coerce").iloc[0]
        gap = f", top-2 gap {round(float(gap_value), 2)} m"

    if group_name == "strong":
        return f"Successful: {label}, facade distance {distance} m, {facade_count} candidate edges."
    if group_name == "ambiguous":
        return f"Difficult: multiple close facades{gap}, {facade_count} candidate edges."
    if group_name == "improved_after_reranking":
        return f"Before {label}; after {row.get('new_confidence_label', '')}, rank gap {row.get('new_rank_gap_m', '')} m."
    if group_name == "corner_ambiguity":
        return f"Corner ambiguity: POI lies near a facade endpoint{gap}."
    if group_name == "multi_tenant":
        return f"Multi-tenant proxy: {row.get('_pois_in_building', '')} POIs share this building."
    return str(label)


def run():
    if gpd is None:
        raise ImportError("geopandas is required for visualization output.")

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    buildings = load_buildings()
    buildings_m, metric_crs = to_metric_crs(buildings)
    streets_m = load_streets(metric_crs)
    example_groups = pick_examples(matches, buildings_m, streets_m)

    captions = []
    for group_name, group_df in example_groups.items():
        if group_df is None or group_df.empty:
            note(f"No examples available for {group_name}.")
            continue
        for i, (_, row) in enumerate(group_df.head(MAX_PER_GROUP).iterrows(), start=1):
            result = plot_example(row, group_name, i, buildings_m, streets_m)
            if result is None:
                note(f"Skipped {group_name} example {i}; required geometry was unavailable.")
                continue
            output_path, caption = result
            captions.append(
                {
                    "group": group_name,
                    "image_path": str(output_path),
                    "poi_id": row.get("poi_id", ""),
                    "matched_building_id": row.get("matched_building_id", ""),
                    "confidence_label": row.get("confidence_label", ""),
                    "caption": caption,
                }
            )
            print(f"Wrote {output_path}")

    pd.DataFrame(captions).to_csv(CAPTIONS_CSV, index=False)
    print(f"Wrote captions: {CAPTIONS_CSV}")


if __name__ == "__main__":
    run()
