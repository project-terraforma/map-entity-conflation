from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
import pandas as pd
from shapely import wkb

from .config import PipelineConfig
from .preprocessing import canonicalize_layer
from .utils import ensure_directory

LOGGER = logging.getLogger(__name__)


def load_study_area(path: str | Path) -> gpd.GeoDataFrame:
    """Read a study area polygon or multipolygon from GeoJSON or another vector format."""
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Study area at {path} is empty.")
    return gdf[["geometry"]].copy()


def clip_to_study_area(gdf: gpd.GeoDataFrame, polygon_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatially clip a layer to the study polygon."""
    if gdf.empty:
        return gdf.copy()
    clipped = gpd.clip(gdf, polygon_gdf)
    return clipped.reset_index(drop=True)


def _is_remote_uri(path: str | Path) -> bool:
    return isinstance(path, str) and "://" in path


def _study_area_bounds_4326(study_area: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = study_area.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _duckdb_escape(value: str) -> str:
    return value.replace("'", "''")


def _remote_path_candidates(path: str) -> list[str]:
    candidates = [path]
    if path.endswith("/*"):
        candidates.append(f"{path[:-1]}*.parquet")
    if "/type=" in path and path.endswith("/*.parquet"):
        theme_prefix = path.split("/type=", 1)[0]
        candidates.append(f"{theme_prefix}/*/*.parquet")
    elif "/type=" in path and path.endswith("/*"):
        theme_prefix = path.split("/type=", 1)[0]
        candidates.append(f"{theme_prefix}/*/*.parquet")
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _flatten_mapping(value: Any, prefix: str) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if not isinstance(value, dict):
        return flattened
    for key, nested_value in value.items():
        nested_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(nested_value, dict):
            flattened.update(_flatten_mapping(nested_value, nested_key))
        else:
            flattened[nested_key] = nested_value
    return flattened


def _flatten_struct_columns(df: pd.DataFrame) -> pd.DataFrame:
    flattened = df.copy()
    new_columns: dict[str, list[Any]] = {}

    for column in list(flattened.columns):
        series = flattened[column]
        if not series.map(lambda value: isinstance(value, dict)).any():
            continue
        for index, value in series.items():
            nested_items = _flatten_mapping(value, column) if isinstance(value, dict) else {}
            for nested_key, nested_value in nested_items.items():
                new_columns.setdefault(nested_key, [None] * len(flattened))
                new_columns[nested_key][index] = nested_value

    for column, values in new_columns.items():
        flattened[column] = values

    return flattened


def _build_remote_bbox_sql(columns: list[str], bounds_4326: tuple[float, float, float, float]) -> str | None:
    minx, miny, maxx, maxy = bounds_4326
    column_set = set(columns)

    if "bbox" in column_set:
        return (
            f"bbox.xmax >= {minx} AND bbox.xmin <= {maxx} "
            f"AND bbox.ymax >= {miny} AND bbox.ymin <= {maxy}"
        )

    direct_bbox_columns = {"bbox.xmin", "bbox.ymin", "bbox.xmax", "bbox.ymax"}
    if direct_bbox_columns.issubset(column_set):
        return (
            f'"bbox.xmax" >= {minx} AND "bbox.xmin" <= {maxx} '
            f'AND "bbox.ymax" >= {miny} AND "bbox.ymin" <= {maxy}'
        )

    xmin_aliases = ["xmin", "minx", "x_min", "bbox_xmin"]
    ymin_aliases = ["ymin", "miny", "y_min", "bbox_ymin"]
    xmax_aliases = ["xmax", "maxx", "x_max", "bbox_xmax"]
    ymax_aliases = ["ymax", "maxy", "y_max", "bbox_ymax"]

    def _first_match(candidates: list[str]) -> str | None:
        return next((candidate for candidate in candidates if candidate in column_set), None)

    xmin_col = _first_match(xmin_aliases)
    ymin_col = _first_match(ymin_aliases)
    xmax_col = _first_match(xmax_aliases)
    ymax_col = _first_match(ymax_aliases)
    if xmin_col and ymin_col and xmax_col and ymax_col:
        return (
            f'"{xmax_col}" >= {minx} AND "{xmin_col}" <= {maxx} '
            f'AND "{ymax_col}" >= {miny} AND "{ymin_col}" <= {maxy}'
        )

    return None


def _coerce_geometry(value: Any):
    if value is None:
        return None
    if hasattr(value, "geom_type"):
        return value
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        return wkb.loads(value)
    raise TypeError(f"Unsupported geometry value type: {type(value)!r}")


def _configure_duckdb_for_overture(connection: duckdb.DuckDBPyConnection) -> None:
    try:
        connection.execute("INSTALL httpfs")
    except Exception as exc:
        LOGGER.debug("DuckDB INSTALL httpfs skipped or failed: %s", exc)

    try:
        connection.execute("LOAD httpfs")
    except Exception as exc:
        raise RuntimeError(
            "DuckDB could not load the httpfs extension required for reading Overture from S3. "
            "If this is your first run, DuckDB may need network access to install extensions. "
            "You can also try running `INSTALL httpfs; LOAD httpfs;` in DuckDB manually."
        ) from exc

    connection.execute("SET s3_region='us-west-2'")


def _read_remote_parquet(path: str, study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    bounds_4326 = _study_area_bounds_4326(study_area)
    connection = duckdb.connect()
    try:
        _configure_duckdb_for_overture(connection)
        attempted_paths = _remote_path_candidates(path)
        schema = None
        selected_path = None
        last_error: Exception | None = None

        for candidate_path in attempted_paths:
            escaped_path = _duckdb_escape(candidate_path)
            try:
                schema = connection.execute(
                    "DESCRIBE SELECT * FROM read_parquet("
                    f"'{escaped_path}', filename=true, hive_partitioning=1)"
                ).df()
                selected_path = candidate_path
                break
            except duckdb.IOException as exc:
                last_error = exc
                LOGGER.debug("Remote parquet path probe failed for %s: %s", candidate_path, exc)

        if schema is None or selected_path is None:
            tried = ", ".join(attempted_paths)
            raise RuntimeError(
                "DuckDB could not find Overture parquet files at the configured S3 path. "
                f"Tried: {tried}. "
                "Check the release version and theme/type path."
            ) from last_error

        columns = schema["column_name"].astype(str).tolist()
        bbox_filter = _build_remote_bbox_sql(columns, bounds_4326)
        where_clause = f" WHERE {bbox_filter}" if bbox_filter else ""
        escaped_selected_path = _duckdb_escape(selected_path)
        query = (
            "SELECT * FROM read_parquet("
            f"'{escaped_selected_path}', filename=true, hive_partitioning=1)"
            f"{where_clause}"
        )
        LOGGER.info("Reading remote parquet for study area bounds %s from %s", bounds_4326, selected_path)
        df = connection.execute(query).df()
    finally:
        connection.close()

    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")

    df = _flatten_struct_columns(df)
    if "geometry" not in df.columns:
        raise ValueError(f"Remote parquet did not contain a geometry column: {path}")
    geometry = df["geometry"].map(_coerce_geometry)
    return gpd.GeoDataFrame(df.drop(columns=["geometry"]), geometry=geometry, crs="EPSG:4326")


def _read_parquet_glob(pattern: str) -> gpd.GeoDataFrame:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No parquet files matched pattern: {pattern}")
    frames = [gpd.read_parquet(match) for match in matches]
    combined = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=frames[0].crs)


def _read_vector(path: str | Path, study_area: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    if _is_remote_uri(path):
        if study_area is None:
            raise ValueError("study_area is required when reading remote parquet inputs.")
        return _read_remote_parquet(str(path), study_area)

    if isinstance(path, str) and any(token in path for token in ["*", "?", "["]):
        suffix = Path(path).suffix.lower()
        if suffix in {".parquet", ".geoparquet"}:
            return _read_parquet_glob(path)
        raise ValueError(f"Wildcard input paths are only supported for parquet-based inputs: {path}")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".geoparquet"}:
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def _load_and_prepare(
    path: str | Path | None,
    study_area: gpd.GeoDataFrame,
    layer_name: str,
    *,
    id_candidates: list[str],
    name_candidates: list[str] | None = None,
    number_candidates: list[str] | None = None,
    street_candidates: list[str] | None = None,
    road_class_candidates: list[str] | None = None,
    interim_dir: Path | None = None,
    write_interim: bool = False,
) -> gpd.GeoDataFrame:
    if path is None:
        LOGGER.warning("No path configured for %s; returning empty GeoDataFrame.", layer_name)
        return gpd.GeoDataFrame(geometry=[], crs=study_area.crs)

    gdf = _read_vector(path, study_area=study_area)
    if gdf.crs is None:
        LOGGER.warning("%s layer has no CRS; assuming EPSG:4326.", layer_name)
        gdf = gdf.set_crs("EPSG:4326")
    if study_area.crs is not None and gdf.crs != study_area.crs:
        gdf = gdf.to_crs(study_area.crs)

    clipped = clip_to_study_area(gdf, study_area)
    prepared = canonicalize_layer(
        clipped,
        layer_name,
        id_candidates=id_candidates,
        name_candidates=name_candidates,
        number_candidates=number_candidates,
        street_candidates=street_candidates,
        road_class_candidates=road_class_candidates,
    )

    if write_interim and interim_dir is not None:
        ensure_directory(interim_dir)
        output_path = interim_dir / f"{layer_name}_clipped.parquet"
        prepared.to_parquet(output_path, index=False)
        LOGGER.info("Wrote clipped %s extract to %s", layer_name, output_path)

    return prepared


def load_overture_places(config: PipelineConfig, study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return _load_and_prepare(
        config.resolve_path(config.get("inputs.places_path")),
        study_area,
        "places",
        id_candidates=["id", "feature_id", "overture_id", "gers_id"],
        name_candidates=["name", "primary_name", "names.primary", "common_name", "feature_name"],
        number_candidates=["house_number", "number", "addr:housenumber"],
        street_candidates=["street", "street_name", "addr_street", "address_street"],
        interim_dir=config.interim_dir,
        write_interim=bool(config.get("inputs.write_clipped_extracts", True)),
    )


def load_overture_buildings(config: PipelineConfig, study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return _load_and_prepare(
        config.resolve_path(config.get("inputs.buildings_path")),
        study_area,
        "buildings",
        id_candidates=["id", "feature_id", "overture_id"],
        street_candidates=["street", "street_name", "primary_street"],
        interim_dir=config.interim_dir,
        write_interim=bool(config.get("inputs.write_clipped_extracts", True)),
    )


def load_overture_addresses(config: PipelineConfig, study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return _load_and_prepare(
        config.resolve_path(config.get("inputs.addresses_path")),
        study_area,
        "addresses",
        id_candidates=["id", "feature_id", "overture_id"],
        number_candidates=["house_number", "number", "addr:housenumber"],
        street_candidates=["street", "street_name", "road_name", "addr:street"],
        interim_dir=config.interim_dir,
        write_interim=bool(config.get("inputs.write_clipped_extracts", True)),
    )


def load_overture_roads(config: PipelineConfig, study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return _load_and_prepare(
        config.resolve_path(config.get("inputs.roads_path")),
        study_area,
        "roads",
        id_candidates=["id", "feature_id", "overture_id", "segment_id"],
        name_candidates=["name", "street_name", "road_name"],
        street_candidates=["name", "street_name", "road_name"],
        road_class_candidates=["class", "road_class", "subtype", "kind"],
        interim_dir=config.interim_dir,
        write_interim=bool(config.get("inputs.write_clipped_extracts", True)),
    )


def load_reference_csvs(paths: list[str | Path] | None) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in paths:
        resolved = Path(path)
        if not resolved.exists():
            LOGGER.warning("Reference CSV not found: %s", resolved)
            continue
        frame = pd.read_csv(resolved)
        frame["source_csv"] = str(resolved)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
