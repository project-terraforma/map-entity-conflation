"""Create POI/building GeoDataFrames and attach the best available building match."""
import pandas as pd
from shapely.geometry import Point

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from config import BUILDING_NEAREST_FALLBACK_THRESHOLD_M
from data_loader import first_existing, parse_geometry, stringify_value
from geometry_utils import to_metric_crs


def _series_or_empty(df, column):
    if column and column in df.columns:
        return df[column]
    return pd.Series([""] * len(df), index=df.index)


def standardize_places(places):
    """Build a minimal POI table from whatever place columns are actually present."""
    if places is None or places.empty:
        return None, ["places layer missing or empty"]
    df = places.copy()
    geom_col = first_existing(df, ["geometry", "geom", "wkb_geometry", "geometry_wkt"])
    lat_col = first_existing(df, ["poi_lat", "latitude", "lat", "y"])
    lon_col = first_existing(df, ["poi_lon", "longitude", "lon", "lng", "x"])
    id_col = first_existing(df, ["poi_id", "id", "place_id", "overture_id"])
    name_col = first_existing(df, ["poi_name", "name", "names"])
    category_col = first_existing(df, ["poi_category", "category", "categories", "primary_category", "types", "poi_types"])
    building_col = first_existing(df, ["matched_building_id", "overture_building_id", "building_id", "building_ids", "building"])

    notes = []
    if id_col is None:
        notes.append("places id column not found; using row index as poi_id")
    if geom_col is None and (lat_col is None or lon_col is None):
        notes.append("places geometry/lat/lon not found")
        return None, notes

    out = pd.DataFrame(index=df.index)
    out["poi_id"] = _series_or_empty(df, id_col).apply(stringify_value) if id_col else df.index.astype(str)
    out["poi_name"] = _series_or_empty(df, name_col).apply(stringify_value)
    out["poi_category"] = _series_or_empty(df, category_col).apply(stringify_value)
    out["source_building_id"] = _series_or_empty(df, building_col).apply(stringify_value)

    if geom_col:
        out["geometry"] = df[geom_col].apply(parse_geometry)
    else:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        out["geometry"] = [Point(x, y) if pd.notna(x) and pd.notna(y) else None for x, y in zip(lon, lat)]
    out = out[out["geometry"].apply(lambda g: g is not None and not g.is_empty and g.geom_type == "Point")].copy()
    if out.empty:
        notes.append("no valid point geometries could be built for places")
        return None, notes
    return gpd.GeoDataFrame(out, geometry="geometry", crs=getattr(places, "crs", None) or "EPSG:4326"), notes


def standardize_buildings(buildings):
    """Build a minimal building footprint table from actual columns."""
    if buildings is None or buildings.empty:
        return None, ["buildings layer missing or empty"]
    df = buildings.copy()
    geom_col = first_existing(df, ["geometry", "geom", "wkb_geometry", "geometry_wkt"])
    id_col = first_existing(df, ["building_id", "id", "overture_id"])
    notes = []
    if geom_col is None:
        return None, ["buildings geometry column not found"]
    if id_col is None:
        notes.append("building id column not found; using row index as building_id")
    out = pd.DataFrame(index=df.index)
    out["building_id"] = _series_or_empty(df, id_col).apply(stringify_value) if id_col else df.index.astype(str)
    out["geometry"] = df[geom_col].apply(parse_geometry)
    out = out[out["geometry"].apply(lambda g: g is not None and not g.is_empty and g.geom_type in {"Polygon", "MultiPolygon"})].copy()
    if out.empty:
        notes.append("no valid polygon building footprints found")
        return None, notes
    return gpd.GeoDataFrame(out, geometry="geometry", crs=getattr(buildings, "crs", None) or "EPSG:4326"), notes


def standardize_points(df, layer_name, id_aliases):
    """Standardize optional point-like layers such as addresses or entrances."""
    if df is None or df.empty:
        return None, [f"{layer_name} layer missing or empty"]
    geom_col = first_existing(df, ["geometry", "geom", "wkb_geometry", "geometry_wkt"])
    lat_col = first_existing(df, ["latitude", "lat", "y", f"{layer_name}_lat"])
    lon_col = first_existing(df, ["longitude", "lon", "lng", "x", f"{layer_name}_lon"])
    id_col = first_existing(df, id_aliases + ["id", "overture_id"])
    if geom_col is None and (lat_col is None or lon_col is None):
        return None, [f"{layer_name} geometry/lat/lon not found"]
    out = pd.DataFrame(index=df.index)
    out[f"{layer_name}_id"] = _series_or_empty(df, id_col).apply(stringify_value) if id_col else df.index.astype(str)
    if geom_col:
        out["geometry"] = df[geom_col].apply(parse_geometry)
    else:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        out["geometry"] = [Point(x, y) if pd.notna(x) and pd.notna(y) else None for x, y in zip(lon, lat)]
    out = out[out["geometry"].apply(lambda g: g is not None and not g.is_empty and g.geom_type == "Point")].copy()
    if out.empty:
        return None, [f"no valid point geometries found for {layer_name}"]
    return gpd.GeoDataFrame(out, geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326"), []


def standardize_lines(df, layer_name):
    """Standardize optional line layers such as streets/segments."""
    if df is None or df.empty:
        return None, [f"{layer_name} layer missing or empty"]
    geom_col = first_existing(df, ["geometry", "geom", "wkb_geometry", "geometry_wkt"])
    id_col = first_existing(df, ["street_segment_id", "segment_id", "id", "overture_id"])
    name_col = first_existing(df, ["street_name", "name", "names", "primary_name"])
    if geom_col is None:
        return None, [f"{layer_name} geometry column not found"]
    out = pd.DataFrame(index=df.index)
    out["street_segment_id"] = _series_or_empty(df, id_col).apply(stringify_value) if id_col else df.index.astype(str)
    out["street_name"] = _series_or_empty(df, name_col).apply(stringify_value)
    out["geometry"] = df[geom_col].apply(parse_geometry)
    out = out[out["geometry"].apply(lambda g: g is not None and not g.is_empty and g.geom_type in {"LineString", "MultiLineString"})].copy()
    if out.empty:
        return None, [f"no valid line geometries found for {layer_name}"]
    return gpd.GeoDataFrame(out, geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326"), []


def apply_problem1_matches(poi_gdf, problem1_output):
    """Attach matched_building_id from Problem 1 output when matching POI ids exist."""
    if poi_gdf is None:
        return poi_gdf, ["cannot apply Problem 1 output without POIs"]
    if problem1_output is None or problem1_output.empty:
        poi_gdf["problem1_building_id"] = ""
        return poi_gdf, ["Problem 1 output unavailable"]
    poi_id_col = first_existing(problem1_output, ["poi_id", "id", "place_id"])
    building_col = first_existing(problem1_output, ["matched_building_id", "poi_building_geom_id", "address_building_geom_id"])
    if poi_id_col is None or building_col is None:
        poi_gdf["problem1_building_id"] = ""
        return poi_gdf, ["Problem 1 output missing poi_id or matched building column"]
    lookup = (
        problem1_output[[poi_id_col, building_col]]
        .dropna()
        .drop_duplicates(subset=[poi_id_col])
        .assign(**{poi_id_col: lambda d: d[poi_id_col].apply(stringify_value), building_col: lambda d: d[building_col].apply(stringify_value)})
        .set_index(poi_id_col)[building_col]
        .to_dict()
    )
    poi_gdf["problem1_building_id"] = poi_gdf["poi_id"].map(lookup).fillna("")
    return poi_gdf, [f"Problem 1 building ids attached for {(poi_gdf['problem1_building_id'] != '').sum()} POIs"]


def link_pois_to_buildings(layers):
    """Return projected POIs/buildings/optional layers with POI-level building ids."""
    if gpd is None:
        raise ImportError("geopandas is required for Problem 2. Install requirements.txt.")

    poi_gdf, poi_notes = standardize_places(layers.get("places"))
    b_gdf, building_notes = standardize_buildings(layers.get("buildings"))
    address_gdf, address_notes = standardize_points(layers.get("addresses"), "address", ["address_id"])
    entrance_gdf, entrance_notes = standardize_points(layers.get("entrances"), "entrance", ["entrance_id", "connector_id"])
    streets_gdf, street_notes = standardize_lines(layers.get("streets"), "streets")
    notes = poi_notes + building_notes + address_notes + entrance_notes + street_notes

    poi_gdf, p1_notes = apply_problem1_matches(poi_gdf, layers.get("problem1_output"))
    notes.extend(p1_notes)

    if poi_gdf is None or b_gdf is None:
        return None, None, None, None, notes

    poi_m, metric_crs = to_metric_crs(poi_gdf)
    b_m = b_gdf.to_crs(metric_crs)
    address_m = address_gdf.to_crs(metric_crs) if address_gdf is not None else None
    entrance_m = entrance_gdf.to_crs(metric_crs) if entrance_gdf is not None else None
    streets_m = streets_gdf.to_crs(metric_crs) if streets_gdf is not None else None

    building_lookup = b_m.drop_duplicates("building_id").set_index("building_id").geometry.to_dict()
    poi_m["matched_building_id"] = poi_m["problem1_building_id"].where(poi_m["problem1_building_id"] != "", poi_m["source_building_id"])
    poi_m["building_match_method"] = ""
    poi_m.loc[poi_m["problem1_building_id"] != "", "building_match_method"] = "problem1_output"
    poi_m.loc[(poi_m["building_match_method"] == "") & (poi_m["source_building_id"] != ""), "building_match_method"] = "place_building_column"

    missing_known_geom = ~poi_m["matched_building_id"].isin(building_lookup.keys())
    needs_spatial = (poi_m["matched_building_id"] == "") | missing_known_geom
    if needs_spatial.any():
        try:
            nearest = gpd.sjoin_nearest(
                poi_m.loc[needs_spatial].copy(),
                b_m[["building_id", "geometry"]],
                how="left",
                max_distance=BUILDING_NEAREST_FALLBACK_THRESHOLD_M,
                distance_col="distance_to_building_m",
            )
            nearest = nearest.dropna(subset=["building_id"]).drop_duplicates(subset=["poi_id"])
            nearest_lookup = nearest.set_index("poi_id")["building_id"].to_dict()
            nearest_distance = nearest.set_index("poi_id")["distance_to_building_m"].to_dict()
            fill_mask = poi_m["poi_id"].isin(nearest_lookup)
            poi_m.loc[fill_mask, "matched_building_id"] = poi_m.loc[fill_mask, "poi_id"].map(nearest_lookup)
            poi_m.loc[fill_mask, "building_match_method"] = "spatial_nearest_building"
            poi_m["distance_to_building_m"] = poi_m["poi_id"].map(nearest_distance)
        except Exception as exc:
            notes.append(f"spatial nearest building fallback failed: {exc}")

    poi_m["has_building_match"] = poi_m["matched_building_id"].isin(building_lookup.keys())
    notes.append(f"POIs with usable building footprint match: {int(poi_m['has_building_match'].sum())} of {len(poi_m)}")
    return poi_m, b_m, entrance_m, streets_m, notes
