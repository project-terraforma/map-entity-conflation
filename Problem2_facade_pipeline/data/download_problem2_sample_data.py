"""Download small Overture sample extracts for Problem 2 facade matching.

The script uses the official ``overturemaps`` CLI when available and writes
GeoJSON files with names already expected by ``problem2_facade_matching/config.py``.
It does not fabricate entrance/connectors data; optional layers are skipped when
the Overture client cannot download them for the configured bounding boxes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None


DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"

REGIONS = {
    "boulder": {
        "label": "Boulder Pearl Street Mall",
        "bbox": (-105.2865, 40.0150, -105.2760, 40.0215),
    },
    "louisville": {
        "label": "Louisville, Colorado downtown",
        "bbox": (-105.1336847, 39.9750387, -105.127107, 39.9814959),
    },
}

LAYERS = {
    "places": {
        "overture_type": "place",
        "output": "places.geojson",
        "required": True,
    },
    "buildings": {
        "overture_type": "building",
        "output": "buildings.geojson",
        "required": True,
    },
    "addresses": {
        "overture_type": "address",
        "output": "addresses.geojson",
        "required": False,
    },
    "streets": {
        "overture_type": "segment",
        "output": "segments.geojson",
        "required": False,
    },
}


def bbox_arg(region_name: str) -> str:
    """Return one bbox string for one configured small sample region."""
    west, south, east, north = REGIONS[region_name]["bbox"]
    return f"{west},{south},{east},{north}"


def get_overture_command() -> list[str]:
    """Return an executable command prefix for the Overture CLI."""
    executable = shutil.which("overturemaps")
    if executable:
        return [executable]
    return [sys.executable, "-m", "overturemaps"]


def merge_geojson_parts(part_paths: list[Path], output_path: Path) -> None:
    """Merge per-region GeoJSON files into the single filename expected by config.py."""
    existing_parts = [path for path in part_paths if path.exists()]
    if not existing_parts:
        return
    if gpd is not None:
        frames = [gpd.read_file(path) for path in existing_parts]
        merged = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            geometry="geometry",
            crs=frames[0].crs,
        )
        merged.to_file(output_path, driver="GeoJSON")
        return

    features = []
    for path in existing_parts:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        features.extend(data.get("features", []))
    with output_path.open("w", encoding="utf-8") as file:
        json.dump({"type": "FeatureCollection", "features": features}, file)


def download_layer(layer_name: str, layer: dict[str, object], region_names: list[str], overwrite: bool) -> tuple[Path, bool]:
    """Download one layer; return output path and whether the download succeeded."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / str(layer["output"])
    if output_path.exists() and not overwrite:
        print(f"Keeping existing {layer_name}: {output_path}")
        return output_path, True

    part_paths = []
    success = True
    for region_name in region_names:
        part_path = RAW_DIR / f"{output_path.stem}_{region_name}{output_path.suffix}"
        part_paths.append(part_path)
        command = get_overture_command() + [
            "download",
            f"--bbox={bbox_arg(region_name)}",
            "-f",
            "geojson",
            f"--type={layer['overture_type']}",
            "-o",
            str(part_path),
        ]
        if part_path.exists() and overwrite:
            part_path.unlink()
        print(f"Downloading {layer_name} for {REGIONS[region_name]['label']} to {part_path}")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(command, text=True, capture_output=True, env=env, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            success = False
            if part_path.exists():
                part_path.unlink()
            print(f"Skipped {layer_name} for {region_name}: overturemaps download failed")
            if result.stderr.strip():
                print(result.stderr.strip())
            elif result.stdout.strip():
                print(result.stdout.strip())

    if output_path.exists():
        output_path.unlink()
    if success:
        try:
            merge_geojson_parts(part_paths, output_path)
        except Exception as exc:
            print(f"Skipped {layer_name}: downloaded parts could not be merged: {exc}")
            if output_path.exists():
                output_path.unlink()
            return output_path, False
        return output_path, output_path.exists()
    return output_path, False


def geometry_summary(path: Path) -> tuple[int, list[str], str, str]:
    """Return row count, columns, geometry column name, and geometry types."""
    if gpd is not None:
        gdf = gpd.read_file(path)
        geom_col = str(gdf.geometry.name) if gdf.geometry is not None else ""
        geom_types = ",".join(sorted(str(value) for value in gdf.geometry.geom_type.dropna().unique()))
        return len(gdf), [str(col) for col in gdf.columns], geom_col, geom_types

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    features = data.get("features", [])
    columns = set()
    geom_types = set()
    for feature in features:
        columns.update((feature.get("properties") or {}).keys())
        geometry = feature.get("geometry") or {}
        if geometry.get("type"):
            geom_types.add(str(geometry["type"]))
    columns.add("geometry")
    return len(features), sorted(columns), "geometry", ",".join(sorted(geom_types))


def print_file_report(layer_name: str, path: Path, success: bool) -> None:
    """Print exact columns and geometry info for one downloaded file."""
    print("")
    print(f"layer: {layer_name}")
    print(f"file path: {path}")
    if not success or not path.exists():
        print("row count: 0")
        print("columns: []")
        print("geometry column/type: missing")
        return
    try:
        row_count, columns, geom_col, geom_types = geometry_summary(path)
    except Exception as exc:
        print(f"row count: unable to read file: {exc}")
        print("columns: []")
        print("geometry column/type: unreadable")
        return
    print(f"row count: {row_count}")
    print(f"columns: {columns}")
    print(f"geometry column/type: {geom_col} / {geom_types}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=sorted(REGIONS),
        default=sorted(REGIONS),
        help="Small sample regions to include in one combined extract.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing sample files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = ", ".join(REGIONS[name]["label"] for name in args.regions)
    print(f"Problem 2 sample regions: {labels}")
    for name in args.regions:
        print(f"{REGIONS[name]['label']} bbox: {bbox_arg(name)}")

    required_failures = []
    for layer_name, layer in LAYERS.items():
        path, success = download_layer(layer_name, layer, args.regions, args.overwrite)
        print_file_report(layer_name, path, success)
        if layer["required"] and not success:
            required_failures.append(layer_name)

    if required_failures:
        print("")
        print(f"Required downloads failed: {required_failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
