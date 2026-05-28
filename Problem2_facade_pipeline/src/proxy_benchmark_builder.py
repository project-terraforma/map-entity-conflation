"""Build proxy storefront candidate points from Boulder/Louisville sign data.

The source files contain sign and entrance/front-door coordinates, not verified
facade labels. This module cleans those point observations into a small CSV
that can be joined to Problem 2 facade predictions for proxy evaluation.
"""

from pathlib import Path

import pandas as pd

from config import BENCHMARK_DIR, PROXY_CANDIDATES_OUTPUT, PROXY_SOURCE_DIR_OPTIONS, PROXY_SOURCE_FILES, REPO_ROOT


TYPE_MAP = {
    "sign": "sign",
    "window_sign": "sign",
    "front_door": "front_door",
    "front door": "front_door",
    "fornt_door": "front_door",
    "front_)door": "front_door",
    "rear_door": "entrance",
    "back_door": "entrance",
}


REQUIRED_COLUMNS = {
    "POI_name",
    "overture_id",
    "image_filename",
    "Type",
    "Remarks",
    "Latitude",
    "Longitude",
    "Exclude",
}


def _source_file_paths():
    """Return existing source CSV paths keyed by dataset name."""
    found = {}
    for dataset, filename in PROXY_SOURCE_FILES.items():
        for source_dir in PROXY_SOURCE_DIR_OPTIONS:
            path = source_dir / filename
            if path.exists():
                found[dataset] = path
                break
    return found


def _clean_text(value):
    """Return a safe string value for CSV output."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _display_path(path):
    """Use repo-relative paths in generated benchmark CSVs when possible."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _normalize_type(value):
    """Map dataset Type values to proxy point categories."""
    raw = _clean_text(value).lower().strip()
    return TYPE_MAP.get(raw, raw)


def _load_one(dataset, path: Path):
    """Load and clean one source dataset."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    rows = []
    for source_row_number, row in df.iterrows():
        exclude = pd.to_numeric(pd.Series([row.get("Exclude")]), errors="coerce").iloc[0]
        if pd.notna(exclude) and int(exclude) != 0:
            continue

        lat = pd.to_numeric(pd.Series([row.get("Latitude")]), errors="coerce").iloc[0]
        lon = pd.to_numeric(pd.Series([row.get("Longitude")]), errors="coerce").iloc[0]
        if pd.isna(lat) or pd.isna(lon):
            continue

        proxy_type = _normalize_type(row.get("Type"))
        if proxy_type not in {"sign", "front_door", "entrance"}:
            continue

        overture_id = _clean_text(row.get("overture_id"))
        rows.append(
            {
                "source_dataset": dataset,
                "source_file": _display_path(path),
                "source_row_number": int(source_row_number) + 2,
                "poi_id": overture_id,
                "overture_poi_id": overture_id,
                "poi_name": _clean_text(row.get("POI_name")),
                "proxy_point_type": proxy_type,
                "proxy_lat": float(lat),
                "proxy_lon": float(lon),
                "image_reference": _clean_text(row.get("image_filename")),
                "notes": _clean_text(row.get("Remarks")),
            }
        )
    return pd.DataFrame(rows), len(df)


def build_proxy_candidates(output_path=PROXY_CANDIDATES_OUTPUT):
    """Build and write the cleaned proxy candidate CSV."""
    source_paths = _source_file_paths()
    if not source_paths:
        searched = "\n".join(str(path / filename) for path in PROXY_SOURCE_DIR_OPTIONS for filename in PROXY_SOURCE_FILES.values())
        print("Proxy benchmark source files were not found. Place Boulder/Louisville CSVs at one of these paths:")
        print(searched)
        return None

    frames = []
    loaded_rows = 0
    for dataset, path in source_paths.items():
        cleaned, raw_count = _load_one(dataset, path)
        loaded_rows += raw_count
        frames.append(cleaned)
        print(f"Loaded proxy source {dataset}: {path} ({raw_count} raw rows, {len(cleaned)} usable rows)")

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote proxy candidates: {output_path} ({len(result)} rows; {loaded_rows} raw rows loaded)")
    return result


def count_raw_proxy_rows():
    """Count raw source rows when source files are available."""
    total = 0
    for path in _source_file_paths().values():
        total += len(pd.read_csv(path))
    return total


if __name__ == "__main__":
    build_proxy_candidates()
