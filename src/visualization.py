from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def plot_study_layers(study_area: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, places: gpd.GeoDataFrame, path: str | Path, figsize: tuple[int, int]) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    study_area.boundary.plot(ax=ax, color="black", linewidth=1.5)
    if not buildings.empty:
        buildings.plot(ax=ax, facecolor="lightgray", edgecolor="dimgray", linewidth=0.4, alpha=0.8)
    if not places.empty:
        places.plot(ax=ax, color="crimson", markersize=12, alpha=0.8)
    ax.set_title("Study area, buildings, and places")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_resolved_links(
    left_gdf: gpd.GeoDataFrame,
    right_gdf: gpd.GeoDataFrame,
    resolved_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    path: str | Path,
    title: str,
    figsize: tuple[int, int],
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    if not right_gdf.empty:
        if right_gdf.geom_type.iloc[0] in {"Polygon", "MultiPolygon"}:
            right_gdf.plot(ax=ax, facecolor="whitesmoke", edgecolor="gray", linewidth=0.5, alpha=0.9)
        else:
            right_gdf.plot(ax=ax, color="steelblue", linewidth=1.0, alpha=0.7)
    if not left_gdf.empty:
        left_gdf.plot(ax=ax, color="darkorange", markersize=10, alpha=0.8)

    if not resolved_df.empty:
        left_lookup = left_gdf.set_index(left_key)["geometry"]
        right_lookup = right_gdf.set_index(right_key)["geometry"]
        for _, row in resolved_df[resolved_df["is_resolved"]].iterrows():
            left_geom = left_lookup.get(row[left_key])
            right_geom = right_lookup.get(row[right_key])
            if left_geom is None or right_geom is None:
                continue
            left_point = left_geom.centroid if left_geom.geom_type != "Point" else left_geom
            right_point = right_geom.centroid if right_geom.geom_type != "Point" else right_geom
            ax.plot([left_point.x, right_point.x], [left_point.y, right_point.y], color="black", linewidth=0.6, alpha=0.5)

    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_unresolved_cases(base_gdf: gpd.GeoDataFrame, unresolved_gdf: gpd.GeoDataFrame, path: str | Path, title: str, figsize: tuple[int, int]) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    if not base_gdf.empty:
        if base_gdf.geom_type.iloc[0] in {"Polygon", "MultiPolygon"}:
            base_gdf.plot(ax=ax, facecolor="lightgray", edgecolor="gray", linewidth=0.4, alpha=0.7)
        else:
            base_gdf.plot(ax=ax, color="lightgray", linewidth=1.0, alpha=0.7)
    if not unresolved_gdf.empty:
        unresolved_gdf.plot(ax=ax, color="red", markersize=18, alpha=0.9)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
