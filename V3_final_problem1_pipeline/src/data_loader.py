"""Load and standardize raw Overture layers for Problem 1."""

from pathlib import Path

import pandas as pd
from shapely import wkb, wkt
from shapely.geometry import Point, box

from config import (
    ADDRESSES_INPUT_OPTIONS,
    BBOX,
    BUILDINGS_INPUT_OPTIONS,
    ENABLE_BOUNDING_BOX,
    ENABLE_SAMPLING,
    PLACES_INPUT_OPTIONS,
    RAW_DATA_DIR,
    SAMPLE_SIZE,
    STREETS_INPUT_OPTIONS,
)
from normalization import build_address_text, is_missing, stringify_value

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - dependency is listed, fallback remains useful.
    gpd = None


def find_input_file(options, layer_name, required=True):
    """Return the preferred available raw file for a layer."""
    for option in options:
        path = RAW_DATA_DIR / option
        if path.exists():
            return path
    if required:
        expected = " or ".join(options[:3])
        raise FileNotFoundError(
            f"Error: No {layer_name} file found in {RAW_DATA_DIR}. "
            f"Please provide {expected}."
        )
    return None


def load_table(path: Path) -> pd.DataFrame:
    """Load a CSV, GeoJSON, or Parquet file."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".geojson", ".json"}:
        if gpd is None:
            raise ImportError("GeoJSON input requires geopandas. Install requirements.txt.")
        return gpd.read_file(path)
    raise ValueError(f"Unsupported input format for {path}")


def parse_geometry(value):
    """Decode shapely, WKB bytes/hex, or WKT geometry values."""
    if is_missing(value):
        return None
    if hasattr(value, "geom_type"):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return wkb.loads(bytes(value))
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.upper().startswith(("POINT", "POLYGON", "MULTI", "LINESTRING")):
                return wkt.loads(text)
            return wkb.loads(bytes.fromhex(text))
        except Exception:
            return None
    return None


def first_existing(df, names):
    """Return the first matching column name from a list of aliases."""
    lower_map = {str(col).lower(): col for col in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def get_value(row, names, default=None):
    """Read the first present, non-empty value from a row."""
    for name in names:
        if name in row.index and not is_missing(row.get(name)):
            return row.get(name)
    return default


def extract_lat_lon(df, lat_aliases, lon_aliases, geometry_col="geometry"):
    """Extract latitude/longitude from columns or point geometry."""
    lat_col = first_existing(df, lat_aliases)
    lon_col = first_existing(df, lon_aliases)
    lat = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else pd.Series([pd.NA] * len(df), index=df.index)
    lon = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else pd.Series([pd.NA] * len(df), index=df.index)

    if geometry_col in df.columns and (lat.isna().any() or lon.isna().any()):
        geom = df[geometry_col].apply(parse_geometry)
        geom_lat = geom.apply(lambda g: g.y if getattr(g, "geom_type", None) == "Point" else pd.NA)
        geom_lon = geom.apply(lambda g: g.x if getattr(g, "geom_type", None) == "Point" else pd.NA)
        lat = lat.fillna(geom_lat)
        lon = lon.fillna(geom_lon)
    return lat, lon


def extract_place_address(row):
    """Extract a readable POI address from common Overture place fields."""
    address_value = get_value(row, ["address", "addresses", "freeform", "addr:full"])
    if isinstance(address_value, list) and address_value:
        address_value = address_value[0]
    if isinstance(address_value, dict):
        text = build_address_text(address_value)
        if text:
            return text
    return stringify_value(address_value)


def extract_building_id(row):
    """Extract an Overture building id attached to a place when present."""
    value = get_value(row, ["overture_building_id", "building_id", "building_ids", "building"])
    if isinstance(value, (list, tuple)) and value:
        return stringify_value(value[0])
    return stringify_value(value)


def standardize_places(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize raw places to V3 POI columns."""
    if df.empty:
        raise ValueError("Error: Places input is empty.")

    lat, lon = extract_lat_lon(df, ["poi_lat", "latitude", "lat", "y"], ["poi_lon", "longitude", "lon", "lng", "x"])
    id_col = first_existing(df, ["poi_id", "id", "place_id", "overture_id"])
    name_col = first_existing(df, ["poi_name", "name", "names"])
    types_col = first_existing(df, ["poi_types", "types", "categories", "category", "primary_category"])

    if id_col is None:
        raise ValueError("Error: Places input is missing an id column such as id or poi_id.")

    result = pd.DataFrame(index=df.index)
    result["poi_id"] = df[id_col].apply(stringify_value)
    result["poi_name"] = df[name_col].apply(stringify_value) if name_col else ""
    result["poi_lat"] = lat
    result["poi_lon"] = lon
    result["poi_address_input"] = df.apply(extract_place_address, axis=1)
    result["poi_types"] = df[types_col].apply(stringify_value) if types_col else ""
    result["overture_building_id"] = df.apply(extract_building_id, axis=1)
    return result


def standardize_addresses(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize raw addresses to V3 address columns."""
    if df.empty:
        raise ValueError("Error: Addresses input is empty.")

    lat, lon = extract_lat_lon(df, ["address_lat", "latitude", "lat", "y"], ["address_lon", "longitude", "lon", "lng", "x"])
    id_col = first_existing(df, ["address_id", "id", "overture_id"])
    street_col = first_existing(df, ["address_street", "street", "street_name"])
    number_col = first_existing(df, ["address_number", "number", "house_number", "housenumber"])

    if id_col is None:
        raise ValueError("Error: Addresses input is missing an id column such as id or address_id.")

    result = pd.DataFrame(index=df.index)
    result["address_id"] = df[id_col].apply(stringify_value)
    result["address_street"] = df[street_col].apply(stringify_value) if street_col else ""
    result["address_number"] = df[number_col].apply(stringify_value) if number_col else ""
    result["address_text"] = df.apply(build_address_text, axis=1)
    result["address_lat"] = lat
    result["address_lon"] = lon
    return result


def standardize_buildings(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize raw buildings to V3 building columns."""
    if df.empty:
        raise ValueError("Error: Buildings input is empty.")
    id_col = first_existing(df, ["building_id", "id", "overture_id"])
    geom_col = first_existing(df, ["geometry", "geom", "wkb_geometry", "geometry_wkt"])
    if id_col is None:
        raise ValueError("Error: Buildings input is missing an id column such as id or building_id.")

    result = pd.DataFrame(index=df.index)
    result["building_id"] = df[id_col].apply(stringify_value)
    if geom_col:
        result["geometry"] = df[geom_col].apply(parse_geometry)
    else:
        lat, lon = extract_lat_lon(df, ["latitude", "lat", "y"], ["longitude", "lon", "lng", "x"], geometry_col="")
        result["geometry"] = [Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else None for xy in zip(lon, lat)]
    return result


def standardize_streets(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize optional raw street segments."""
    id_col = first_existing(df, ["street_segment_id", "segment_id", "id", "overture_id"])
    name_col = first_existing(df, ["street_name", "name", "names", "primary_name"])
    geom_col = first_existing(df, ["geometry", "geom", "wkb_geometry", "geometry_wkt"])
    result = pd.DataFrame(index=df.index)
    result["street_segment_id"] = df[id_col].apply(stringify_value) if id_col else df.index.astype(str)
    result["street_name"] = df[name_col].apply(stringify_value) if name_col else ""
    result["geometry"] = df[geom_col].apply(parse_geometry) if geom_col else None
    return result


def apply_bbox(df, lat_col=None, lon_col=None, geometry_col=None):
    """Apply optional bounding box filtering."""
    if not ENABLE_BOUNDING_BOX:
        return df
    min_lon, min_lat, max_lon, max_lat = BBOX
    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        return df[
            df[lat_col].between(min_lat, max_lat, inclusive="both")
            & df[lon_col].between(min_lon, max_lon, inclusive="both")
        ].copy()
    if geometry_col and geometry_col in df.columns:
        bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
        return df[df[geometry_col].apply(lambda g: g is not None and g.intersects(bbox_geom))].copy()
    return df


def load_raw_layers():
    """Load all required raw layers and optional street data."""
    places_path = find_input_file(PLACES_INPUT_OPTIONS, "places")
    addresses_path = find_input_file(ADDRESSES_INPUT_OPTIONS, "addresses")
    buildings_path = find_input_file(BUILDINGS_INPUT_OPTIONS, "buildings")
    streets_path = find_input_file(STREETS_INPUT_OPTIONS, "streets/segments", required=False)

    places = standardize_places(load_table(places_path))
    addresses = standardize_addresses(load_table(addresses_path))
    buildings = standardize_buildings(load_table(buildings_path))
    streets = standardize_streets(load_table(streets_path)) if streets_path else None

    places = apply_bbox(places, "poi_lat", "poi_lon")
    addresses = apply_bbox(addresses, "address_lat", "address_lon")
    buildings = apply_bbox(buildings, geometry_col="geometry")
    if streets is not None:
        streets = apply_bbox(streets, geometry_col="geometry")

    if ENABLE_SAMPLING and len(places) > SAMPLE_SIZE:
        places = places.sample(n=SAMPLE_SIZE, random_state=42).copy()

    for layer_name, df in [("POIs", places), ("Addresses", addresses), ("Buildings", buildings)]:
        if df.empty:
            raise ValueError(f"Error: {layer_name} dataset is empty after filtering.")

    print(f"Loaded POIs: {len(places)}")
    print(f"Loaded Addresses: {len(addresses)}")
    print(f"Loaded Buildings: {len(buildings)}")
    if streets is not None:
        print(f"Loaded Street Segments: {len(streets)}")
    else:
        print("Loaded Street Segments: 0 (optional layer not provided)")

    return places, addresses, buildings, streets
