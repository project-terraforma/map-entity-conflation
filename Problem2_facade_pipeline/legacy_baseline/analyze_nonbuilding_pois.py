"""Analyze how likely non-building POIs affect Problem 2 facade ambiguity.

This script uses only fields that exist in the downloaded Overture places data.
It does not modify the facade matching algorithm.
"""
from pathlib import Path

import pandas as pd

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLACES_PATH = PROJECT_ROOT / "Problem2_facade_pipeline" / "data" / "raw" / "places.geojson"
MATCHES_PATH = PROJECT_ROOT / "outputs" / "problem2_facade_matches.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "problem2_nonbuilding_analysis.csv"

AMBIGUOUS_LABEL = "needs_review_multiple_close_facades"
NO_BUILDING_LABEL = "needs_review_no_building_match"

NONBUILDING_BASIC_CATEGORIES = {
    "gas_station",
    "historic_site",
    "monument",
    "national_park",
    "park",
    "parking",
    "playground",
    "public_plaza",
    "public_transit_facility_or_service",
    "recreational_trail_or_path",
}

NONBUILDING_TEXT_PATTERNS = [
    " park",
    "parking",
    "playground",
    "plaza",
    "trail",
    "transit",
    "bus_station",
    "gas_station",
    "monument",
    "historic_site",
    "national_park",
]


def note(message):
    print(f"NOTE: {message}")


def load_places():
    if not PLACES_PATH.exists():
        raise FileNotFoundError(f"Downloaded Overture places file not found: {PLACES_PATH}")
    if gpd is None:
        raise ImportError("geopandas is required to read the downloaded places GeoJSON.")
    places = gpd.read_file(PLACES_PATH)
    print(f"Loaded places: {PLACES_PATH} ({len(places)} rows)")
    print(f"Places columns: {list(places.columns)}")
    return places


def load_matches():
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(f"Problem 2 match output not found: {MATCHES_PATH}")
    matches = pd.read_csv(MATCHES_PATH, low_memory=False)
    print(f"Loaded facade matches: {MATCHES_PATH} ({len(matches)} rows)")
    print(f"Match columns: {list(matches.columns)}")
    return matches


def normalize_text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).lower().replace("-", "_")


def classify_place(row):
    """Return likely non-building classification from existing Overture place fields."""
    reasons = []

    basic = normalize_text(row.get("basic_category"))
    if basic in NONBUILDING_BASIC_CATEGORIES:
        reasons.append(f"basic_category={basic}")

    for column in ["categories", "taxonomy"]:
        if column not in row.index:
            continue
        text = f" {normalize_text(row.get(column))} "
        for pattern in NONBUILDING_TEXT_PATTERNS:
            if pattern in text:
                reasons.append(f"{column}_contains={pattern.strip()}")
                break

    return bool(reasons), ";".join(dict.fromkeys(reasons))


def inspect_categories(places):
    if "basic_category" not in places.columns:
        note("basic_category column is missing; classifier will rely on categories/taxonomy text only.")
        return pd.DataFrame()

    counts = places["basic_category"].fillna("").astype(str).value_counts().reset_index()
    counts.columns = ["basic_category", "place_count"]
    counts["matches_nonbuilding_basic_category"] = counts["basic_category"].isin(NONBUILDING_BASIC_CATEGORIES)

    print("\nObserved basic_category values matching the non-building classifier:")
    matched = counts[counts["matches_nonbuilding_basic_category"]]
    if matched.empty:
        print("(none)")
    else:
        print(matched.to_string(index=False))
    return counts


def join_places_to_matches(places, matches):
    required_place_cols = ["id"]
    missing_place = [column for column in required_place_cols if column not in places.columns]
    if missing_place:
        raise ValueError(f"Cannot join places to matches; missing place columns: {missing_place}")
    if "poi_id" not in matches.columns:
        raise ValueError("Cannot join places to matches; problem2_facade_matches.csv has no poi_id column.")

    classifier_results = places.apply(classify_place, axis=1)
    place_flags = pd.DataFrame(
        {
            "poi_id": places["id"].astype(str),
            "basic_category": places["basic_category"].astype(str) if "basic_category" in places.columns else "",
            "likely_nonbuilding_poi": [item[0] for item in classifier_results],
            "nonbuilding_reason": [item[1] for item in classifier_results],
        }
    )
    return matches.merge(place_flags, on="poi_id", how="left")


def summarize_group(df, group_name, mask):
    subset = df[mask].copy()
    total = len(subset)
    row = {
        "section": "group_summary",
        "group": group_name,
        "count": total,
        "percent_of_all": round((total / len(df)) * 100, 2) if len(df) else 0.0,
        "ambiguity_count": pd.NA,
        "ambiguity_rate_percent": pd.NA,
        "no_building_count": pd.NA,
        "no_building_rate_percent": pd.NA,
        "detail": "",
    }
    if "confidence_label" not in subset.columns:
        row["detail"] = "confidence_label missing; rates unavailable"
        return row

    labels = subset["confidence_label"].fillna("").astype(str)
    ambiguity_count = int(labels.eq(AMBIGUOUS_LABEL).sum())
    no_building_count = int(labels.eq(NO_BUILDING_LABEL).sum())
    row["ambiguity_count"] = ambiguity_count
    row["ambiguity_rate_percent"] = round((ambiguity_count / total) * 100, 2) if total else 0.0
    row["no_building_count"] = no_building_count
    row["no_building_rate_percent"] = round((no_building_count / total) * 100, 2) if total else 0.0
    return row


def category_breakdown(joined):
    rows = []
    if "basic_category" not in joined.columns:
        note("basic_category is unavailable after join; category breakdown skipped.")
        return rows

    nonbuilding = joined[joined["likely_nonbuilding_poi"].fillna(False)].copy()
    if nonbuilding.empty:
        return rows

    grouped = nonbuilding.groupby("basic_category", dropna=False)
    for basic_category, group in grouped:
        labels = group["confidence_label"].fillna("").astype(str) if "confidence_label" in group.columns else pd.Series([], dtype=str)
        count = len(group)
        ambiguity_count = int(labels.eq(AMBIGUOUS_LABEL).sum()) if not labels.empty else pd.NA
        no_building_count = int(labels.eq(NO_BUILDING_LABEL).sum()) if not labels.empty else pd.NA
        rows.append(
            {
                "section": "likely_nonbuilding_category",
                "group": basic_category,
                "count": count,
                "percent_of_all": round((count / len(joined)) * 100, 2) if len(joined) else 0.0,
                "ambiguity_count": ambiguity_count,
                "ambiguity_rate_percent": round((ambiguity_count / count) * 100, 2) if count and ambiguity_count is not pd.NA else pd.NA,
                "no_building_count": no_building_count,
                "no_building_rate_percent": round((no_building_count / count) * 100, 2) if count and no_building_count is not pd.NA else pd.NA,
                "detail": "basic_category",
            }
        )
    return rows


def before_after_rows(joined):
    labels = joined["confidence_label"].fillna("").astype(str)
    total_before = len(joined)
    ambiguity_before = int(labels.eq(AMBIGUOUS_LABEL).sum())
    no_building_before = int(labels.eq(NO_BUILDING_LABEL).sum())

    filtered = joined[~joined["likely_nonbuilding_poi"].fillna(False)].copy()
    filtered_labels = filtered["confidence_label"].fillna("").astype(str)
    total_after = len(filtered)
    ambiguity_after = int(filtered_labels.eq(AMBIGUOUS_LABEL).sum())
    no_building_after = int(filtered_labels.eq(NO_BUILDING_LABEL).sum())

    return [
        {
            "section": "before_after_filtering",
            "group": "before_filter_all_pois",
            "count": total_before,
            "percent_of_all": 100.0,
            "ambiguity_count": ambiguity_before,
            "ambiguity_rate_percent": round((ambiguity_before / total_before) * 100, 2) if total_before else 0.0,
            "no_building_count": no_building_before,
            "no_building_rate_percent": round((no_building_before / total_before) * 100, 2) if total_before else 0.0,
            "detail": "",
        },
        {
            "section": "before_after_filtering",
            "group": "after_filter_building_style_pois",
            "count": total_after,
            "percent_of_all": round((total_after / total_before) * 100, 2) if total_before else 0.0,
            "ambiguity_count": ambiguity_after,
            "ambiguity_rate_percent": round((ambiguity_after / total_after) * 100, 2) if total_after else 0.0,
            "no_building_count": no_building_after,
            "no_building_rate_percent": round((no_building_after / total_after) * 100, 2) if total_after else 0.0,
            "detail": "likely_nonbuilding_poi excluded",
        },
    ]


def print_findings(rows):
    df = pd.DataFrame(rows)
    groups = df[df["section"] == "group_summary"].set_index("group")
    before_after = df[df["section"] == "before_after_filtering"].set_index("group")

    print("\nNon-building POI impact:")
    for group in ["all_pois", "likely_nonbuilding_pois", "building_style_pois"]:
        if group in groups.index:
            row = groups.loc[group]
            print(
                f"- {group}: {int(row['count'])} POIs; "
                f"ambiguity {row['ambiguity_rate_percent']}%; "
                f"no-building {row['no_building_rate_percent']}%"
            )

    if {"before_filter_all_pois", "after_filter_building_style_pois"}.issubset(before_after.index):
        before = before_after.loc["before_filter_all_pois"]
        after = before_after.loc["after_filter_building_style_pois"]
        print(
            f"- Ambiguity before filtering: {before['ambiguity_rate_percent']}%; "
            f"after filtering: {after['ambiguity_rate_percent']}%"
        )
        print(
            f"- No-building-match before filtering: {before['no_building_rate_percent']}%; "
            f"after filtering: {after['no_building_rate_percent']}%"
        )


def run():
    places = load_places()
    matches = load_matches()
    inspect_categories(places)
    joined = join_places_to_matches(places, matches)

    if joined["likely_nonbuilding_poi"].isna().any():
        note("some Problem 2 POIs did not join to places data; treating unmatched rows as building-style for filtering.")
        joined["likely_nonbuilding_poi"] = joined["likely_nonbuilding_poi"].fillna(False)
        joined["nonbuilding_reason"] = joined["nonbuilding_reason"].fillna("")

    all_mask = pd.Series([True] * len(joined), index=joined.index)
    nonbuilding_mask = joined["likely_nonbuilding_poi"].fillna(False)
    building_style_mask = ~nonbuilding_mask

    rows = [
        summarize_group(joined, "all_pois", all_mask),
        summarize_group(joined, "likely_nonbuilding_pois", nonbuilding_mask),
        summarize_group(joined, "building_style_pois", building_style_mask),
    ]
    rows.extend(before_after_rows(joined))
    rows.extend(category_breakdown(joined))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print_findings(rows)
    print(f"\nWrote non-building analysis: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
