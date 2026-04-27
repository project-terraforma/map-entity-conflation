from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .candidate_generation import (
    generate_address_building_candidates,
    generate_building_street_candidates,
    generate_place_address_candidates,
    generate_place_building_candidates,
)
from .conflation_bundle import build_conflation_bundle
from .config import PipelineConfig
from .evaluation import build_error_samples, compare_place_references, summarize_resolution_table, write_metrics_report
from .feature_engineering import (
    compute_address_building_features,
    compute_building_street_features,
    compute_place_address_features,
    compute_place_building_features,
    estimate_entrance_points,
    prepare_layer_keys,
)
from .io_overture import (
    load_overture_addresses,
    load_overture_buildings,
    load_overture_places,
    load_overture_roads,
    load_reference_csvs,
    load_study_area,
)
from .preprocessing import ensure_projected_crs
from .resolution import resolve_best_matches
from .scoring import score_address_building, score_building_street, score_place_address, score_place_building
from .utils import ensure_directory, setup_logging
from .visualization import plot_resolved_links, plot_study_layers, plot_unresolved_cases

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Project Zephr baseline conflation pipeline.")
    parser.add_argument("--config", required=True, help="Path to the YAML settings file.")
    return parser.parse_args()


def _write_tabular_outputs(df: pd.DataFrame, base_path: Path) -> None:
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    df.to_parquet(base_path.with_suffix(".parquet"), index=False)


def _write_geoparquet_outputs(df: pd.DataFrame, geometry_column: str, crs: str | None, base_path: Path) -> None:
    if df.empty or geometry_column not in df.columns:
        return
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry_column, crs=crs)
    gdf.to_parquet(base_path.with_name(f"{base_path.stem}_geo.parquet"), index=False)


def run_pipeline(config_path: str | Path) -> None:
    setup_logging()
    config = PipelineConfig.from_yaml(config_path)
    config.require_input_paths(
        [
            "study_area_path",
            "inputs.places_path",
            "inputs.buildings_path",
            "inputs.addresses_path",
            "inputs.roads_path",
        ]
    )
    ensure_directory(config.interim_dir)
    ensure_directory(config.output_dir)
    ensure_directory(config.figures_dir)

    study_area = load_study_area(config.resolve_path(config.get("study_area_path")))
    study_area = ensure_projected_crs(study_area, config.target_crs)

    places = ensure_projected_crs(load_overture_places(config, study_area.to_crs("EPSG:4326")), config.target_crs)
    buildings = ensure_projected_crs(load_overture_buildings(config, study_area.to_crs("EPSG:4326")), config.target_crs)
    addresses = ensure_projected_crs(load_overture_addresses(config, study_area.to_crs("EPSG:4326")), config.target_crs)
    roads = ensure_projected_crs(load_overture_roads(config, study_area.to_crs("EPSG:4326")), config.target_crs)
    places, buildings, addresses, roads = prepare_layer_keys(places, buildings, addresses, roads)

    reference_csvs = [config.resolve_path(path) for path in config.get("reference_csv_paths", [])]
    reference_df = load_reference_csvs(reference_csvs)

    ab_cfg = config.get("candidate_generation.address_building", {})
    pb_cfg = config.get("candidate_generation.place_building", {})
    bs_cfg = config.get("candidate_generation.building_street", {})
    pa_cfg = config.get("candidate_generation.place_address", {})

    address_building_candidates = generate_address_building_candidates(
        addresses,
        buildings,
        max_distance_m=float(ab_cfg.get("max_distance_m", 60.0)),
        nearest_n=int(ab_cfg.get("nearest_n", 3)),
    )
    place_building_candidates = generate_place_building_candidates(
        places,
        buildings,
        max_distance_m=float(pb_cfg.get("max_distance_m", 80.0)),
        nearest_n=int(pb_cfg.get("nearest_n", 3)),
    )

    address_building_features = compute_address_building_features(address_building_candidates, addresses, buildings)
    address_building_scored = score_address_building(address_building_features, config.get("scoring.address_building", {}))
    address_building_resolved = resolve_best_matches(
        address_building_scored,
        "address_id",
        "score",
        threshold=float(config.get("scoring.address_building.threshold", 0.45)),
        min_score_gap=float(config.get("resolution.min_score_gap", 0.1)),
    )

    place_building_features = compute_place_building_features(
        place_building_candidates,
        places,
        buildings,
        address_building_resolved[address_building_resolved["is_resolved"]],
        addresses,
    )
    place_building_scored = score_place_building(place_building_features, config.get("scoring.place_building", {}))
    place_building_resolved = resolve_best_matches(
        place_building_scored,
        "place_id",
        "score",
        threshold=float(config.get("scoring.place_building.threshold", 0.42)),
        min_score_gap=float(config.get("resolution.min_score_gap", 0.1)),
    )

    building_street_candidates = generate_building_street_candidates(
        buildings,
        roads,
        max_distance_m=float(bs_cfg.get("max_distance_m", 100.0)),
        nearest_n=int(bs_cfg.get("nearest_n", 3)),
    )
    building_street_features = compute_building_street_features(
        building_street_candidates,
        buildings,
        roads,
        address_building_resolved[address_building_resolved["is_resolved"]],
        addresses,
        frontage_buffer_m=float(bs_cfg.get("frontage_buffer_m", 15.0)),
    )
    building_street_scored = score_building_street(building_street_features, config.get("scoring.building_street", {}))
    building_street_resolved = resolve_best_matches(
        building_street_scored,
        "building_id",
        "score",
        threshold=float(config.get("scoring.building_street.threshold", 0.4)),
        min_score_gap=float(config.get("resolution.min_score_gap", 0.1)),
    )

    place_address_candidates = generate_place_address_candidates(
        places,
        addresses,
        place_building_resolved[place_building_resolved["is_resolved"]],
        address_building_resolved[address_building_resolved["is_resolved"]],
        max_distance_m=float(pa_cfg.get("max_distance_m", 80.0)),
        nearest_n=int(pa_cfg.get("nearest_n", 3)),
    )
    place_address_features = compute_place_address_features(
        place_address_candidates,
        places,
        addresses,
        place_building_resolved[place_building_resolved["is_resolved"]],
        address_building_resolved[address_building_resolved["is_resolved"]],
    )
    place_address_scored = score_place_address(place_address_features, config.get("scoring.place_address", {}))
    place_address_resolved = resolve_best_matches(
        place_address_scored,
        "place_id",
        "score",
        threshold=float(config.get("scoring.place_address.threshold", 0.4)),
        min_score_gap=float(config.get("resolution.min_score_gap", 0.1)),
    )

    entrance_points = estimate_entrance_points(
        building_street_resolved[building_street_resolved["is_resolved"]],
        buildings,
        roads,
    )
    building_street_resolved = building_street_resolved.merge(entrance_points, on=["building_id", "street_segment_id"], how="left")

    address_to_building = address_building_resolved[
        ["address_id", "building_id", "score", "confidence_tier", "match_reason", "is_resolved"]
    ].copy()
    place_to_building = place_building_resolved[
        ["place_id", "place_name", "building_id", "score", "confidence_tier", "match_reason", "is_resolved"]
    ].copy()
    building_to_street = building_street_resolved[
        [
            "building_id",
            "street_segment_id",
            "street_name",
            "score",
            "confidence_tier",
            "match_reason",
            "is_resolved",
            "estimated_entrance_geometry",
        ]
    ].copy()
    place_to_address = place_address_resolved[
        ["place_id", "address_id", "score", "confidence_tier", "match_reason", "is_resolved"]
    ].copy()
    conflation_bundle = build_conflation_bundle(
        place_to_building,
        place_to_address,
        address_to_building,
        building_to_street,
    )

    _write_tabular_outputs(address_to_building, config.output_dir / "address_to_building")
    _write_tabular_outputs(place_to_building, config.output_dir / "place_to_building")
    _write_tabular_outputs(building_to_street, config.output_dir / "building_to_street")
    _write_tabular_outputs(place_to_address, config.output_dir / "place_to_address")
    _write_tabular_outputs(conflation_bundle, config.output_dir / "conflation_bundle")

    _write_geoparquet_outputs(
        address_building_resolved.merge(addresses[["address_id", "geometry"]], on="address_id", how="left"),
        "geometry",
        str(addresses.crs),
        config.output_dir / "address_to_building",
    )
    _write_geoparquet_outputs(
        place_building_resolved.merge(places[["place_id", "geometry"]], on="place_id", how="left"),
        "geometry",
        str(places.crs),
        config.output_dir / "place_to_building",
    )
    _write_geoparquet_outputs(
        building_street_resolved,
        "estimated_entrance_geometry",
        str(buildings.crs),
        config.output_dir / "building_to_street",
    )
    _write_geoparquet_outputs(
        place_address_resolved.merge(places[["place_id", "geometry"]], on="place_id", how="left"),
        "geometry",
        str(places.crs),
        config.output_dir / "place_to_address",
    )

    figsize_cfg = tuple(config.get("visualization.figsize", [12, 10]))
    plot_study_layers(study_area, buildings, places, config.figures_dir / "study_layers.png", figsize=figsize_cfg)
    plot_resolved_links(
        places,
        buildings,
        place_building_resolved,
        "place_id",
        "building_id",
        config.figures_dir / "place_to_building_links.png",
        "Resolved place to building links",
        figsize=figsize_cfg,
    )
    plot_resolved_links(
        buildings.set_geometry("centroid_geometry"),
        roads,
        building_street_resolved.rename(columns={"street_segment_id": "road_id"}),
        "building_id",
        "road_id",
        config.figures_dir / "building_to_street_links.png",
        "Resolved building to street links",
        figsize=figsize_cfg,
    )
    unresolved_places = place_building_resolved[~place_building_resolved["is_resolved"]].merge(
        places[["place_id", "geometry"]],
        on="place_id",
        how="left",
    )
    unresolved_places = gpd.GeoDataFrame(unresolved_places, geometry="geometry", crs=places.crs)
    plot_unresolved_cases(buildings, unresolved_places, config.figures_dir / "unresolved_places.png", "Unresolved place cases", figsize=figsize_cfg)

    metrics = {
        "address_to_building": summarize_resolution_table(address_to_building, "address_to_building"),
        "place_to_building": summarize_resolution_table(place_to_building, "place_to_building"),
        "building_to_street": summarize_resolution_table(building_to_street, "building_to_street"),
        "place_to_address": summarize_resolution_table(place_to_address, "place_to_address"),
        "conflation_bundle": {
            "rows": int(len(conflation_bundle)),
            "fully_resolved": int(conflation_bundle["is_fully_resolved"].fillna(False).sum()) if not conflation_bundle.empty else 0,
            "partial": int((~conflation_bundle["is_fully_resolved"].fillna(False)).sum()) if not conflation_bundle.empty else 0,
        },
        "reference_comparison": compare_place_references(place_to_building, reference_df),
    }
    write_metrics_report(str(config.output_dir / "metrics_summary.json"), metrics)

    error_samples = build_error_samples(
        {
            "address_to_building": address_to_building,
            "place_to_building": place_to_building,
            "building_to_street": building_to_street,
            "place_to_address": place_to_address,
        }
    )
    error_samples.to_csv(config.output_dir / "error_samples.csv", index=False)

    LOGGER.info("Pipeline complete. Outputs written to %s", config.output_dir)


def main() -> None:
    args = parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
