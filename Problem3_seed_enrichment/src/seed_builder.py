"""Aggregate imagery annotations into one POI seed per observed business."""

import hashlib
from pathlib import Path

import pandas as pd

from config import SOURCE_DIR_OPTIONS, SOURCE_FILES
from text_utils import clean_text, normalize_text

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
FRONT_DOOR_TYPES = {"front_door", "front door", "fornt_door", "front_)door"}


def find_source_dir(explicit_source_dir=None):
    """Return a source directory containing at least one expected CSV."""
    options = [Path(explicit_source_dir)] if explicit_source_dir else SOURCE_DIR_OPTIONS
    for source_dir in options:
        if any((source_dir / filename).exists() for filename in SOURCE_FILES.values()):
            return source_dir
    searched = "\n".join(f"- {path}" for path in options)
    raise FileNotFoundError(f"No Zephr source CSVs found. Looked in:\n{searched}")


def _stable_seed_id(dataset, normalized_name):
    digest = hashlib.sha1(f"{dataset}|{normalized_name}".encode("utf-8")).hexdigest()[:16]
    return f"imagery-{digest}"


def _pipe(values):
    return " | ".join(sorted({clean_text(value) for value in values if clean_text(value)}))


def load_observations(source_dir):
    """Load valid sign and entrance rows from all available source CSVs."""
    frames = []
    for dataset, filename in SOURCE_FILES.items():
        path = Path(source_dir) / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        df = df.copy()
        df["source_dataset"] = dataset
        df["source_file"] = str(path.resolve())
        df["source_row_number"] = df.index + 2
        df["Exclude"] = pd.to_numeric(df["Exclude"], errors="coerce").fillna(0)
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df = df[(df["Exclude"] == 0) & df["Latitude"].notna() & df["Longitude"].notna()].copy()
        df["normalized_name"] = df["POI_name"].apply(normalize_text)
        frames.append(df[df["normalized_name"] != ""])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_seeds(observations, include_existing=False):
    """Return one deterministic seed row per dataset and normalized name."""
    rows = []
    for (dataset, normalized_name), group in observations.groupby(["source_dataset", "normalized_name"], sort=True):
        overture_ids = [clean_text(value) for value in group["overture_id"] if clean_text(value)]
        if overture_ids and not include_existing:
            continue
        group_types = group["Type"].apply(lambda value: clean_text(value).lower())
        preferred = group[group_types.isin(FRONT_DOOR_TYPES)]
        coordinate_rows = preferred if not preferred.empty else group
        rows.append(
            {
                "seed_id": _stable_seed_id(dataset, normalized_name),
                "source_dataset": dataset,
                "poi_name_seed": clean_text(group.iloc[0]["POI_name"]),
                "normalized_name": normalized_name,
                "overture_id_snapshot": _pipe(overture_ids),
                "seed_lat": round(float(coordinate_rows["Latitude"].median()), 7),
                "seed_lon": round(float(coordinate_rows["Longitude"].median()), 7),
                "location_basis": "front_door_median" if not preferred.empty else "observation_median",
                "observation_count": int(len(group)),
                "has_front_door": bool(not preferred.empty),
                "observed_types": _pipe(group["Type"]),
                "image_references": _pipe(group["image_filename"]),
                "remarks": _pipe(group["Remarks"]),
                "source_rows": _pipe(group["source_row_number"]),
                "source_files": _pipe(group["source_file"]),
            }
        )
    return pd.DataFrame(rows)


def build_seeds(explicit_source_dir=None, include_existing=False):
    """Load observations and aggregate imagery-derived POI seeds."""
    source_dir = find_source_dir(explicit_source_dir)
    observations = load_observations(source_dir)
    return aggregate_seeds(observations, include_existing=include_existing), source_dir, len(observations)


def build_seeds_from_raw_extractions(extractions):
    """Convert accepted raw-image OCR rows into the standard seed shape."""
    accepted = extractions[extractions["accepted_as_seed"]].copy()
    if accepted.empty:
        return pd.DataFrame()
    observations = pd.DataFrame(
        {
            "source_dataset": accepted["source_dataset"],
            "POI_name": accepted["poi_name_seed"],
            "overture_id": "",
            "Type": "raw_image",
            "Latitude": accepted["latitude"],
            "Longitude": accepted["longitude"],
            "image_filename": accepted["image_filename"],
            "Remarks": accepted.apply(
                lambda row: f"{row['name_basis']}; confidence={row['extraction_confidence']}; ocr={row['ocr_text']}",
                axis=1,
            ),
            "source_row_number": "",
            "source_file": accepted["image_path"],
            "normalized_name": accepted["poi_name_seed"].apply(normalize_text),
        }
    )
    return aggregate_seeds(observations, include_existing=True)
