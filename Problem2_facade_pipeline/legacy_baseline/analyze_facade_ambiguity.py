"""Analyze why Problem 2 facade ambiguity and failures occur.

This script is analysis-only. It reads existing Problem 2 outputs and available
building geometries, then writes aggregate CSV summaries without changing the
facade matching algorithm.
"""
from pathlib import Path

import pandas as pd
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import nearest_points

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from config import BUILDINGS_INPUT_OPTIONS, OUTPUT_DIR
from data_loader import find_input_file, load_table, parse_geometry
from facade_extractor import extract_facades_from_building
from geometry_utils import to_metric_crs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCHES_CSV = OUTPUT_DIR / "problem2_facade_matches.csv"
DEBUG_SAMPLE_GEOJSON = OUTPUT_DIR / "problem2_debug_sample.geojson"
AMBIGUITY_ANALYSIS_CSV = OUTPUT_DIR / "problem2_ambiguity_analysis.csv"
FAILURE_SUMMARY_CSV = OUTPUT_DIR / "problem2_failure_summary.csv"

AMBIGUOUS_LABEL = "needs_review_multiple_close_facades"
NO_BUILDING_LABEL = "needs_review_no_building_match"
CONFIDENT_LABELS = {
    "high_confidence_nearest_facade",
    "high_confidence_entrance_supported",
    "medium_confidence_street_supported",
}
STREET_SUPPORT_RADIUS_M = 40.0
CORNER_DISTANCE_M = 3.0


def note(message):
    print(f"NOTE: {message}")


def load_matches():
    if not MATCHES_CSV.exists():
        raise FileNotFoundError(f"Required match file not found: {MATCHES_CSV}")
    df = pd.read_csv(MATCHES_CSV, low_memory=False)
    print(f"Loaded matches: {MATCHES_CSV} ({len(df)} rows)")
    print(f"Match columns: {list(df.columns)}")
    return df


def load_buildings():
    if gpd is None:
        note("geopandas is unavailable; building geometry metrics cannot be computed.")
        return None
    path = find_input_file(BUILDINGS_INPUT_OPTIONS, "buildings", required=False)
    if path is None:
        note("building geometry file not found; building size, facade count, and density metrics cannot be computed.")
        return None

    df = load_table(path)
    print(f"Loaded buildings: {path} ({len(df)} rows)")
    print(f"Building columns: {list(df.columns)}")
    if "id" not in df.columns:
        note("building file has no 'id' column; cannot join building geometry to matched_building_id.")
        return None
    if "geometry" not in df.columns:
        note("building file has no 'geometry' column; cannot compute building geometry metrics.")
        return None

    out = df[["id", "geometry"]].copy()
    out["geometry"] = out["geometry"].apply(parse_geometry)
    out = out[out["geometry"].apply(lambda geom: geom is not None and not geom.is_empty)].copy()
    if out.empty:
        note("no valid building geometries found.")
        return None
    return gpd.GeoDataFrame(out.rename(columns={"id": "matched_building_id"}), geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326")


def attach_building_metrics(matches, buildings):
    result = matches.copy()
    metric_columns = [
        "building_area_m2",
        "building_perimeter_m",
        "candidate_facade_count",
        "building_nearest_neighbor_m",
    ]

    if buildings is None:
        for column in metric_columns:
            result[column] = pd.NA
        return result

    buildings_m, metric_crs = to_metric_crs(buildings)
    building_metrics = []
    for _, row in buildings_m.iterrows():
        geom = row.geometry
        facades = extract_facades_from_building(geom)
        building_metrics.append(
            {
                "matched_building_id": str(row["matched_building_id"]),
                "building_area_m2": float(geom.area),
                "building_perimeter_m": float(geom.length),
                "candidate_facade_count": len(facades),
            }
        )
    metrics = pd.DataFrame(building_metrics)

    try:
        nearest = gpd.sjoin_nearest(
            buildings_m[["matched_building_id", "geometry"]].copy(),
            buildings_m[["matched_building_id", "geometry"]].copy(),
            how="left",
            max_distance=100.0,
            distance_col="building_nearest_neighbor_m",
        )
        nearest = nearest[nearest["matched_building_id_left"] != nearest["matched_building_id_right"]]
        nearest_dist = nearest.groupby("matched_building_id_left")["building_nearest_neighbor_m"].min()
        metrics = metrics.merge(
            nearest_dist.rename("building_nearest_neighbor_m"),
            left_on="matched_building_id",
            right_index=True,
            how="left",
        )
    except Exception as exc:
        note(f"building nearest-neighbor density metric could not be computed: {exc}")

    result["matched_building_id"] = result["matched_building_id"].fillna("").astype(str)
    return result.merge(metrics, on="matched_building_id", how="left")


def attach_distance_gap(matches):
    result = matches.copy()
    if {"second_best_facade_distance_m", "distance_to_facade_meters"}.issubset(result.columns):
        first = pd.to_numeric(result["distance_to_facade_meters"], errors="coerce")
        second = pd.to_numeric(result["second_best_facade_distance_m"], errors="coerce")
        result["top2_facade_distance_gap_m"] = second - first
    else:
        result["top2_facade_distance_gap_m"] = pd.NA
        note("top-2 facade distance gap cannot be computed because distance columns are missing.")
    return result


def parse_facade_wkt(value):
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    try:
        return wkt.loads(str(value))
    except Exception:
        return None


def attach_corner_metrics(matches):
    result = matches.copy()
    result["near_selected_facade_corner"] = pd.NA
    result["selected_facade_endpoint_distance_m"] = pd.NA

    required = {"poi_lon", "poi_lat", "facade_wkt"}
    if gpd is None:
        note("geopandas is unavailable; corner-building frequency cannot be computed.")
        return result
    if not required.issubset(result.columns):
        note("corner metric cannot be computed because poi_lon, poi_lat, or facade_wkt is missing.")
        return result

    valid = result.dropna(subset=["poi_lon", "poi_lat", "facade_wkt"]).copy()
    valid["geometry"] = [Point(xy) for xy in zip(pd.to_numeric(valid["poi_lon"], errors="coerce"), pd.to_numeric(valid["poi_lat"], errors="coerce"))]
    valid["facade_geometry"] = valid["facade_wkt"].apply(parse_facade_wkt)
    valid = valid[valid["facade_geometry"].apply(lambda geom: geom is not None and not geom.is_empty)].copy()
    if valid.empty:
        note("corner metric cannot be computed because no valid selected facade WKT was available.")
        return result

    point_gdf = gpd.GeoDataFrame(valid[["geometry"]].copy(), geometry="geometry", crs="EPSG:4326")
    point_m, metric_crs = to_metric_crs(point_gdf)
    facade_m = gpd.GeoSeries(valid["facade_geometry"].values, crs="EPSG:4326").to_crs(metric_crs)

    endpoint_distances = []
    near_corner = []
    for point, line in zip(point_m.geometry, facade_m):
        if line is None or line.is_empty:
            endpoint_distances.append(pd.NA)
            near_corner.append(pd.NA)
            continue
        nearest = nearest_points(point, line)[1]
        coords = list(line.coords)
        endpoint_distance = min(nearest.distance(Point(coords[0])), nearest.distance(Point(coords[-1])))
        endpoint_distances.append(round(float(endpoint_distance), 3))
        near_corner.append(bool(endpoint_distance <= CORNER_DISTANCE_M))

    result.loc[valid.index, "selected_facade_endpoint_distance_m"] = endpoint_distances
    result.loc[valid.index, "near_selected_facade_corner"] = near_corner
    return result


def add_analysis_flags(matches):
    result = matches.copy()
    if "confidence_label" not in result.columns:
        note("confidence_label missing; ambiguity/confidence buckets cannot be identified.")
        result["is_ambiguous"] = False
        result["is_no_building_match"] = False
        result["is_confident"] = False
    else:
        labels = result["confidence_label"].fillna("").astype(str)
        result["is_ambiguous"] = labels.eq(AMBIGUOUS_LABEL)
        result["is_no_building_match"] = labels.eq(NO_BUILDING_LABEL)
        result["is_confident"] = labels.isin(CONFIDENT_LABELS)

    if "street_distance_to_facade_m" in result.columns:
        street_distance = pd.to_numeric(result["street_distance_to_facade_m"], errors="coerce")
        result["has_street_support"] = street_distance.le(STREET_SUPPORT_RADIUS_M)
        result["missing_street_support"] = ~result["has_street_support"]
    else:
        note("street_distance_to_facade_m missing; missing street support cannot be assessed.")
        result["has_street_support"] = pd.NA
        result["missing_street_support"] = pd.NA

    if "matched_building_id" in result.columns:
        tenant_counts = result["matched_building_id"].fillna("").astype(str).value_counts()
        result["pois_in_same_building"] = result["matched_building_id"].fillna("").astype(str).map(tenant_counts)
        result["possible_multi_tenant_building"] = result["pois_in_same_building"].gt(1)
    else:
        note("matched_building_id missing; multi-tenant proxy cannot be computed.")
        result["pois_in_same_building"] = pd.NA
        result["possible_multi_tenant_building"] = pd.NA

    return result


def describe_group(df, mask, group_name):
    subset = df[mask].copy()
    row = {"group": group_name, "poi_count": len(subset)}
    numeric_columns = [
        "candidate_facade_count",
        "top2_facade_distance_gap_m",
        "building_area_m2",
        "building_perimeter_m",
        "building_nearest_neighbor_m",
        "pois_in_same_building",
        "selected_facade_endpoint_distance_m",
    ]
    for column in numeric_columns:
        if column in subset.columns:
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            row[f"{column}_mean"] = round(float(values.mean()), 3) if not values.empty else pd.NA
            row[f"{column}_median"] = round(float(values.median()), 3) if not values.empty else pd.NA

    boolean_columns = [
        "near_selected_facade_corner",
        "has_street_support",
        "missing_street_support",
        "possible_multi_tenant_building",
    ]
    for column in boolean_columns:
        if column in subset.columns:
            values = subset[column].dropna()
            row[f"{column}_count"] = int(values.astype(bool).sum()) if not values.empty else pd.NA
            row[f"{column}_percent"] = round(float(values.astype(bool).mean() * 100), 2) if not values.empty else pd.NA
    return row


def write_ambiguity_analysis(df):
    groups = [
        describe_group(df, df["is_ambiguous"], "ambiguous_multiple_close_facades"),
        describe_group(df, df["is_confident"], "confident_matches"),
        describe_group(df, df["is_no_building_match"], "no_building_match"),
    ]
    out = pd.DataFrame(groups)
    out.to_csv(AMBIGUITY_ANALYSIS_CSV, index=False)
    print(f"Wrote ambiguity analysis: {AMBIGUITY_ANALYSIS_CSV}")
    return out


def write_failure_summary(df):
    rows = []
    total = len(df)

    def bool_series(column):
        if column not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        return df[column].map(lambda value: bool(value) if pd.notna(value) else False)

    buckets = {
        "needs_review_multiple_close_facades": df["is_ambiguous"],
        "no_building_match": df["is_no_building_match"],
        "street_supported": bool_series("has_street_support"),
        "missing_street_support": bool_series("missing_street_support"),
        "near_selected_facade_corner": bool_series("near_selected_facade_corner"),
        "possible_multi_tenant_building": bool_series("possible_multi_tenant_building"),
    }
    for bucket, mask in buckets.items():
        count = int(mask.sum())
        rows.append({"bucket": bucket, "count": count, "percent": round((count / total) * 100, 2) if total else 0.0})

    if "confidence_label" in df.columns:
        for label, count in df["confidence_label"].fillna("").replace("", "(blank)").value_counts().items():
            rows.append({"bucket": f"confidence_label:{label}", "count": int(count), "percent": round((count / total) * 100, 2) if total else 0.0})

    out = pd.DataFrame(rows)
    out.to_csv(FAILURE_SUMMARY_CSV, index=False)
    print(f"Wrote failure summary: {FAILURE_SUMMARY_CSV}")
    return out


def print_findings(df):
    ambiguous = df[df["is_ambiguous"]].copy()
    no_building = df[df["is_no_building_match"]].copy()
    confident = df[df["is_confident"]].copy()

    print("\nConcise findings:")
    if ambiguous.empty:
        print("- No ambiguous POIs found.")
    else:
        facade_count = pd.to_numeric(ambiguous.get("candidate_facade_count"), errors="coerce").dropna()
        gap = pd.to_numeric(ambiguous.get("top2_facade_distance_gap_m"), errors="coerce").dropna()
        corner = ambiguous.get("near_selected_facade_corner")
        street_missing = ambiguous.get("missing_street_support")
        multi = ambiguous.get("possible_multi_tenant_building")
        density = pd.to_numeric(ambiguous.get("building_nearest_neighbor_m"), errors="coerce").dropna()
        confident_area = pd.to_numeric(confident.get("building_area_m2"), errors="coerce").dropna()
        ambiguous_area = pd.to_numeric(ambiguous.get("building_area_m2"), errors="coerce").dropna()

        if not facade_count.empty:
            share_4_plus = round(float(facade_count.ge(4).mean() * 100), 2)
            print(f"- Ambiguous POIs have a median of {round(float(facade_count.median()), 2)} candidate facade edges; {share_4_plus}% are on buildings with 4+ facade edges.")
        else:
            print("- Candidate facade counts could not be computed because building geometries were unavailable or unmatched.")

        if not gap.empty:
            print(f"- Average top-2 facade distance gap for ambiguous POIs is {round(float(gap.mean()), 3)} meters.")
        else:
            print("- Top-2 facade distance gaps could not be computed from the available output columns.")

        if corner is not None and corner.notna().any():
            print(f"- {round(float(corner.dropna().astype(bool).mean() * 100), 2)}% of ambiguous POIs are near the selected facade endpoint, a detectable corner proxy.")
        else:
            print("- Corner frequency could not be computed from selected facade WKT and POI coordinates.")

        if street_missing is not None and street_missing.notna().any():
            print(f"- {round(float(street_missing.dropna().astype(bool).mean() * 100), 2)}% of ambiguous POIs lack street support within {STREET_SUPPORT_RADIUS_M} m.")
        else:
            print("- Missing street support could not be assessed because street distance data is unavailable.")

        if multi is not None and multi.notna().any():
            print(f"- {round(float(multi.dropna().astype(bool).mean() * 100), 2)}% of ambiguous POIs share a building with another POI, a multi-tenant storefront proxy.")
        else:
            print("- Multi-tenant storefront proxy could not be computed because matched building ids are unavailable.")

        if not density.empty:
            print(f"- Ambiguous matched buildings have median nearest-building distance of {round(float(density.median()), 3)} m, a dense-building proxy.")
        else:
            print("- Dense downtown/building proximity could not be computed from available building geometries.")

        if not ambiguous_area.empty and not confident_area.empty:
            print(f"- Median building area is {round(float(ambiguous_area.median()), 1)} m2 for ambiguous matches vs {round(float(confident_area.median()), 1)} m2 for confident matches.")
        else:
            print("- Ambiguous vs confident building size comparison could not be computed.")

    if no_building.empty:
        print("- No-building-match failures were not present.")
    else:
        if "matched_building_id" in no_building.columns:
            blank_ids = no_building["matched_building_id"].fillna("").astype(str).str.strip().eq("").sum()
            print(f"- No-building-match failures: {len(no_building)} total; {blank_ids} have blank matched_building_id values.")
        else:
            print(f"- No-building-match failures: {len(no_building)} total; matched_building_id column is unavailable.")


def maybe_report_debug_sample():
    if DEBUG_SAMPLE_GEOJSON.exists():
        print(f"Debug sample available for visual inspection: {DEBUG_SAMPLE_GEOJSON}")
    else:
        note("problem2_debug_sample.geojson not found; visual sample was not used.")


def run():
    matches = load_matches()
    buildings = load_buildings()
    analyzed = attach_distance_gap(matches)
    analyzed = attach_building_metrics(analyzed, buildings)
    analyzed = attach_corner_metrics(analyzed)
    analyzed = add_analysis_flags(analyzed)

    write_ambiguity_analysis(analyzed)
    write_failure_summary(analyzed)
    print_findings(analyzed)
    maybe_report_debug_sample()


if __name__ == "__main__":
    run()
