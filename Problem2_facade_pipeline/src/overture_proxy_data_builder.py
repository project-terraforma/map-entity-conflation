"""Build Overture extracts for the Boulder/Louisville proxy benchmark areas.

This script derives bounding boxes from the actual proxy sign/entrance points
and downloads Overture layers for those exact areas using the existing
``overturemaps`` CLI style used elsewhere in this repository.
"""

from __future__ import annotations

import argparse
import math
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

from config import (
    PROXY_AREA_ADDRESSES_PATH,
    PROXY_AREA_BUILDINGS_PATH,
    PROXY_AREA_MANIFEST_PATH,
    PROXY_AREA_PLACES_PATH,
    PROXY_AREA_STREETS_PATH,
    PROXY_BBOX_BUFFER_DEGREES,
    PROXY_BBOX_BUFFER_METERS,
    PROXY_OVERTURE_DATA_DIR,
)
from proxy_benchmark_builder import REQUIRED_COLUMNS, _source_file_paths


LAYERS = {
    "places": {"overture_type": "place", "combined_output": PROXY_AREA_PLACES_PATH, "required": True},
    "buildings": {"overture_type": "building", "combined_output": PROXY_AREA_BUILDINGS_PATH, "required": True},
    "addresses": {"overture_type": "address", "combined_output": PROXY_AREA_ADDRESSES_PATH, "required": False},
    "streets": {"overture_type": "segment", "combined_output": PROXY_AREA_STREETS_PATH, "required": False},
}


def get_overture_command() -> list[str]:
    """Return the executable command prefix for the Overture CLI."""
    executable = shutil.which("overturemaps")
    if executable:
        return [executable]
    return [sys.executable, "-m", "overturemaps"]


def detect_coordinate_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Detect latitude/longitude columns from a proxy source dataframe."""
    lower = {str(col).lower(): col for col in df.columns}
    lat_col = lower.get("latitude") or lower.get("lat")
    lon_col = lower.get("longitude") or lower.get("lon") or lower.get("lng")
    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not detect latitude/longitude columns. Columns: {list(df.columns)}")
    return lat_col, lon_col


def load_proxy_source_points() -> pd.DataFrame:
    """Load valid proxy source rows with detected coordinate columns."""
    frames = []
    source_paths = _source_file_paths()
    if not source_paths:
        from config import PROXY_SOURCE_DIR_OPTIONS, PROXY_SOURCE_FILES

        searched = "\n".join(str(path / filename) for path in PROXY_SOURCE_DIR_OPTIONS for filename in PROXY_SOURCE_FILES.values())
        raise FileNotFoundError(f"Proxy source files were not found. Searched: {searched}")

    for dataset, path in source_paths.items():
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        lat_col, lon_col = detect_coordinate_columns(df)
        print(f"{dataset}: detected coordinate columns lat={lat_col}, lon={lon_col}")

        exclude = pd.to_numeric(df["Exclude"], errors="coerce").fillna(0)
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        valid = df[(exclude == 0) & lat.notna() & lon.notna()].copy()
        valid["source_dataset"] = dataset
        valid["proxy_lat"] = lat.loc[valid.index].astype(float)
        valid["proxy_lon"] = lon.loc[valid.index].astype(float)
        frames.append(valid)
        print(f"{dataset}: {len(df)} raw rows, {len(valid)} valid coordinate rows")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def buffer_degrees_for_dataset(df: pd.DataFrame, buffer_degrees: float, buffer_meters: float | None) -> tuple[float, float]:
    """Return longitude/latitude buffer degrees."""
    if buffer_meters is None:
        return buffer_degrees, buffer_degrees
    mean_lat = float(df["proxy_lat"].mean())
    lat_buffer = buffer_meters / 111_320.0
    lon_buffer = buffer_meters / max(1.0, 111_320.0 * abs(math.cos(math.radians(mean_lat))))
    return lon_buffer, lat_buffer


def compute_proxy_bboxes(points: pd.DataFrame, buffer_degrees: float, buffer_meters: float | None) -> dict[str, dict[str, object]]:
    """Compute buffered bboxes by proxy source dataset."""
    bboxes = {}
    for dataset, group in points.groupby("source_dataset"):
        lon_buffer, lat_buffer = buffer_degrees_for_dataset(group, buffer_degrees, buffer_meters)
        west = float(group["proxy_lon"].min()) - lon_buffer
        east = float(group["proxy_lon"].max()) + lon_buffer
        south = float(group["proxy_lat"].min()) - lat_buffer
        north = float(group["proxy_lat"].max()) + lat_buffer
        bboxes[dataset] = {
            "bbox": (west, south, east, north),
            "row_count": int(len(group)),
            "lon_buffer_degrees": lon_buffer,
            "lat_buffer_degrees": lat_buffer,
        }
    return bboxes


def bbox_arg(bbox: tuple[float, float, float, float]) -> str:
    """Format an Overture CLI bbox argument."""
    west, south, east, north = bbox
    return f"{west},{south},{east},{north}"


def download_one(layer_name: str, overture_type: str, dataset: str, bbox: tuple[float, float, float, float], overwrite: bool) -> Path | None:
    """Download one layer for one proxy dataset bbox."""
    output_path = PROXY_OVERTURE_DATA_DIR / f"{dataset}_{layer_name}.parquet"
    if output_path.exists() and not overwrite:
        print(f"Keeping existing {dataset} {layer_name}: {output_path}")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = get_overture_command() + [
        "download",
        f"--bbox={bbox_arg(bbox)}",
        "-f",
        "geoparquet",
        f"--type={overture_type}",
        "-o",
        str(output_path),
    ]
    print(f"Downloading {dataset} {layer_name}: {' '.join(command)}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(command, text=True, capture_output=True, env=env, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        if output_path.exists():
            output_path.unlink()
        print(f"Download failed for {dataset} {layer_name}")
        if result.stderr.strip():
            print(result.stderr.strip())
        elif result.stdout.strip():
            print(result.stdout.strip())
        return None
    return output_path if output_path.exists() else None


def merge_parquet(parts: list[Path], output_path: Path) -> int:
    """Merge parquet parts into one combined parquet output."""
    existing = [path for path in parts if path is not None and path.exists()]
    if not existing:
        return 0
    frames = [pd.read_parquet(path) for path in existing]
    merged = pd.concat(frames, ignore_index=True)
    if "id" in merged.columns:
        merged = merged.drop_duplicates(subset=["id"], keep="first")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    return len(merged)


def write_manifest(bboxes: dict[str, dict[str, object]], layer_rows: list[dict[str, object]]) -> None:
    """Write bbox and output metadata."""
    PROXY_AREA_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, info in bboxes.items():
        west, south, east, north = info["bbox"]
        rows.append(
            {
                "record_type": "bbox",
                "dataset": dataset,
                "layer": "",
                "proxy_rows": info["row_count"],
                "west": west,
                "south": south,
                "east": east,
                "north": north,
                "output_path": "",
                "rows": "",
            }
        )
    rows.extend(layer_rows)
    pd.DataFrame(rows).to_csv(PROXY_AREA_MANIFEST_PATH, index=False)
    print(f"Wrote proxy Overture manifest: {PROXY_AREA_MANIFEST_PATH}")


def build_proxy_overture_data(
    overwrite: bool = False,
    buffer_degrees: float = PROXY_BBOX_BUFFER_DEGREES,
    buffer_meters: float | None = PROXY_BBOX_BUFFER_METERS,
    dry_run: bool = False,
) -> bool:
    """Download proxy-area Overture extracts and write combined parquet files."""
    points = load_proxy_source_points()
    if points.empty:
        print("No valid proxy points found; skipping proxy-area Overture extraction.")
        return False

    bboxes = compute_proxy_bboxes(points, buffer_degrees, buffer_meters)
    for dataset, info in bboxes.items():
        print(f"{dataset}: proxy rows={info['row_count']}, bbox={bbox_arg(info['bbox'])}")

    if dry_run:
        print("Dry run requested; no Overture data was downloaded.")
        write_manifest(bboxes, [])
        return True

    layer_rows = []
    required_failures = []
    for layer_name, layer in LAYERS.items():
        parts = []
        for dataset, info in bboxes.items():
            part = download_one(layer_name, layer["overture_type"], dataset, info["bbox"], overwrite=overwrite)
            if part is not None:
                parts.append(part)
        row_count = merge_parquet(parts, layer["combined_output"])
        success = row_count > 0 and layer["combined_output"].exists()
        print(f"Combined {layer_name}: {layer['combined_output']} ({row_count} rows)")
        layer_rows.append(
            {
                "record_type": "layer",
                "dataset": "combined",
                "layer": layer_name,
                "proxy_rows": "",
                "west": "",
                "south": "",
                "east": "",
                "north": "",
                "output_path": str(layer["combined_output"]),
                "rows": row_count,
            }
        )
        if layer["required"] and not success:
            required_failures.append(layer_name)

    write_manifest(bboxes, layer_rows)
    if required_failures:
        print(f"Required proxy-area Overture layers failed: {required_failures}")
        return False
    return True


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing proxy-area downloads.")
    parser.add_argument("--buffer-degrees", type=float, default=PROXY_BBOX_BUFFER_DEGREES)
    parser.add_argument("--buffer-meters", type=float, default=PROXY_BBOX_BUFFER_METERS)
    parser.add_argument("--dry-run", action="store_true", help="Compute bboxes and write manifest without downloading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return 0 if build_proxy_overture_data(args.overwrite, args.buffer_degrees, args.buffer_meters, args.dry_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
