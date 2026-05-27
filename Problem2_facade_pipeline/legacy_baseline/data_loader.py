"""Load, inspect, and lightly standardize available layers for Problem 2."""
from pathlib import Path

import pandas as pd
from shapely import wkb, wkt

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from config import (
    ADDRESSES_INPUT_OPTIONS,
    BUILDINGS_INPUT_OPTIONS,
    ENTRANCES_INPUT_OPTIONS,
    PLACES_INPUT_OPTIONS,
    PROBLEM1_OUTPUT_DIR,
    PROBLEM1_OUTPUT_OPTIONS,
    PROBLEM1_RAW_DATA_DIR,
    PROBLEM2_RAW_DATA_DIR,
    RAW_DATA_DIR,
    STREETS_INPUT_OPTIONS,
)


def stringify_value(value):
    """Return a compact string for scalar/list/dict values without assuming schema."""
    if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
        return ""
    if isinstance(value, dict):
        for key in ("common", "primary", "name", "value"):
            if key in value:
                return stringify_value(value[key])
        return str(value)
    if isinstance(value, (list, tuple)):
        return "; ".join(stringify_value(v) for v in value[:3])
    return str(value)


def parse_geometry(value):
    """Decode shapely, WKB bytes/hex, or WKT geometry values."""
    if value is None:
        return None
    if not hasattr(value, "geom_type") and not isinstance(value, (bytes, bytearray, memoryview, str)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            return None
    if not hasattr(value, "geom_type") and isinstance(value, float) and pd.isna(value):
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
            if text.upper().startswith(("POINT", "LINESTRING", "POLYGON", "MULTI")):
                return wkt.loads(text)
            return wkb.loads(bytes.fromhex(text))
        except Exception:
            return None
    return None


def first_existing(df, names):
    """Return the first matching column from a case-insensitive alias list."""
    if df is None:
        return None
    lower_map = {str(col).lower(): col for col in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def find_input_file(options, layer_name, required=False):
    """Return the first available layer file across Problem 2, root, and Problem 1 data dirs."""
    search_dirs = [PROBLEM2_RAW_DATA_DIR, RAW_DATA_DIR, PROBLEM1_RAW_DATA_DIR]
    for directory in search_dirs:
        for option in options:
            path = directory / option
            if path.exists():
                return path
    if required:
        expected = ", ".join(options)
        raise FileNotFoundError(f"No {layer_name} file found. Looked for: {expected}")
    return None


def find_problem1_output():
    """Locate a Problem 1 conflation output if one is available."""
    for option in PROBLEM1_OUTPUT_OPTIONS:
        path = PROBLEM1_OUTPUT_DIR / option
        if path.exists():
            return path
    return None


def load_table(path: Path):
    """Load a CSV, GeoJSON, or Parquet file into pandas/geopandas."""
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


def inspect_df(df, name, source_path=None, n=3):
    """Print available columns and sample rows for a dataframe-like object."""
    if df is None:
        print(f"{name}: NOT FOUND")
        return {"layer": name, "source_path": "", "rows": 0, "columns": "", "status": "missing"}
    columns = [str(col) for col in df.columns]
    print(f"\n{name}: source={source_path}")
    print(f"{name}: rows={len(df)}")
    print(f"{name}: columns={columns}")
    try:
        print(f"{name}: sample rows")
        print(df.head(n).to_string())
    except Exception as exc:
        print(f"{name}: unable to print sample rows: {exc}")
    return {
        "layer": name,
        "source_path": str(source_path or ""),
        "rows": len(df),
        "columns": ";".join(columns),
        "status": "loaded",
    }


def load_optional_layer(path, name):
    """Load one optional layer and return (df, inspection_record)."""
    if path is None:
        return None, inspect_df(None, name)
    print(f"Loading {name} from {path}")
    try:
        df = load_table(path)
    except Exception as exc:
        print(f"Failed to load {name}: {exc}")
        return None, {
            "layer": name,
            "source_path": str(path),
            "rows": 0,
            "columns": "",
            "status": f"failed: {exc}",
        }
    return df, inspect_df(df, name, source_path=path)


def load_layers():
    """Load available layers, print columns/samples, and return data plus inspection metadata."""
    layer_specs = [
        ("places", PLACES_INPUT_OPTIONS),
        ("buildings", BUILDINGS_INPUT_OPTIONS),
        ("addresses", ADDRESSES_INPUT_OPTIONS),
        ("entrances", ENTRANCES_INPUT_OPTIONS),
        ("streets", STREETS_INPUT_OPTIONS),
    ]
    layers = {}
    inspection = []
    for name, options in layer_specs:
        path = find_input_file(options, name, required=False)
        df, record = load_optional_layer(path, name)
        layers[name] = df
        inspection.append(record)

    problem1_path = find_problem1_output()
    problem1_df, problem1_record = load_optional_layer(problem1_path, "problem1_output")
    layers["problem1_output"] = problem1_df
    inspection.append(problem1_record)
    return layers, inspection
