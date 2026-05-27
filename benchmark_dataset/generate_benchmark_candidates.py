"""Generate semi-automatic benchmark candidates for Problems 1 and 2.

The generator uses existing repository outputs only. It does not create verified
labels and does not scrape external map data.
"""
from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
PROBLEM1_OUTPUT = ROOT / "Conflation_pipeline" / "outputs" / "final_problem1_conflation.csv"
PROBLEM2_MATCHES = OUTPUTS_DIR / "problem2_facade_matches.csv"
NONBUILDING_ANALYSIS = OUTPUTS_DIR / "problem2_nonbuilding_analysis.csv"
VIS_CAPTIONS = OUTPUTS_DIR / "visualizations" / "captions.csv"

BENCHMARK_CANDIDATES = BENCHMARK_DIR / "benchmark_candidates.csv"
GROUND_TRUTH = BENCHMARK_DIR / "reviewed_ground_truth.csv"
STATISTICS = BENCHMARK_DIR / "benchmark_statistics.csv"
REVIEW_PACKETS = BENCHMARK_DIR / "review_packets"

CANDIDATE_COLUMNS = [
    "poi_id",
    "poi_name",
    "building_id",
    "matched_address",
    "street_name",
    "facade_id",
    "confidence_label",
    "ambiguity_bucket",
    "region",
    "benchmark_type",
    "reviewer_label",
    "reviewer_notes",
]

VIS_PACKET_MAP = {
    "strong": "strong",
    "ambiguous": "ambiguous",
    "improved_after_reranking": "reranked",
    "corner_ambiguity": "corner",
    "multi_tenant": "multi_tenant",
}


def load_csv(path, required=False):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        print(f"Optional file not found: {path}")
        return None
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {path}: {len(df)} rows")
    return df


def infer_region(row):
    lon = pd.to_numeric(pd.Series([row.get("poi_lon")]), errors="coerce").iloc[0] if "poi_lon" in row.index else pd.NA
    lat = pd.to_numeric(pd.Series([row.get("poi_lat")]), errors="coerce").iloc[0] if "poi_lat" in row.index else pd.NA
    if pd.isna(lon) or pd.isna(lat):
        return ""
    if -106.0 <= lon <= -104.5 and 39.5 <= lat <= 40.5:
        return "Boulder Pearl Street Mall"
    if -86.2 <= lon <= -85.3 and 37.9 <= lat <= 38.5:
        return "Louisville downtown"
    return "unknown"


def ambiguity_bucket(row):
    label = str(row.get("confidence_label", "") or "")
    evidence = str(row.get("evidence_fields_used", "") or "")
    if label == "needs_review_no_building_match":
        return "no_building_match"
    if label == "needs_review_multiple_close_facades":
        if "multiple_close_facades" in evidence:
            return "multiple_close_facades"
        return "ambiguous_facade"
    if "street_supported" in label:
        return "street_supported"
    if "nearest_facade" in label:
        return "nearest_facade"
    if "inside_building" in label:
        return "inside_building_nearest_edge"
    return label or "unknown"


def base_candidate(row, benchmark_type):
    return {
        "poi_id": row.get("poi_id", ""),
        "poi_name": row.get("poi_name", ""),
        "building_id": row.get("matched_building_id", ""),
        "matched_address": row.get("matched_address_text", ""),
        "street_name": row.get("matched_street", row.get("nearest_street_name", "")),
        "facade_id": row.get("facade_id", ""),
        "confidence_label": row.get("confidence_label", ""),
        "ambiguity_bucket": ambiguity_bucket(row),
        "region": infer_region(row),
        "benchmark_type": benchmark_type,
        "reviewer_label": "",
        "reviewer_notes": "",
    }


def sample_rows(df, mask, benchmark_type, n=25):
    subset = df[mask].copy()
    if subset.empty:
        return []
    if "poi_id" in subset.columns:
        subset = subset.drop_duplicates("poi_id")
    subset = subset.head(n)
    return [base_candidate(row, benchmark_type) for _, row in subset.iterrows()]


def add_problem1_fields(problem2, problem1):
    result = problem2.copy()
    if problem1 is None or problem1.empty:
        print("Problem 1 output unavailable; matched_address and street_name will remain blank unless present in Problem 2 outputs.")
        return result
    if "poi_id" not in problem1.columns:
        print("Problem 1 output has no poi_id column; skipping Problem 1 join.")
        return result

    join_cols = ["poi_id"]
    for col in ["matched_address_text", "matched_street", "nearest_street_name", "matched_building_id"]:
        if col in problem1.columns:
            join_cols.append(col)
    joined = result.merge(problem1[join_cols].drop_duplicates("poi_id"), on="poi_id", how="left", suffixes=("", "_p1"))
    for source, target in [
        ("matched_address_text", "matched_address_text"),
        ("matched_street", "matched_street"),
        ("nearest_street_name", "nearest_street_name"),
        ("matched_building_id_p1", "problem1_building_id"),
    ]:
        if source in joined.columns and target not in joined.columns:
            joined[target] = joined[source]
    return joined


def build_candidates():
    problem2 = load_csv(PROBLEM2_MATCHES, required=True)
    problem1 = load_csv(PROBLEM1_OUTPUT)
    data = add_problem1_fields(problem2, problem1)

    rows = []
    labels = data["confidence_label"].fillna("").astype(str) if "confidence_label" in data.columns else pd.Series([""] * len(data))
    evidence = data["evidence_fields_used"].fillna("").astype(str) if "evidence_fields_used" in data.columns else pd.Series([""] * len(data))
    facade_ids = data["facade_id"].fillna("").astype(str) if "facade_id" in data.columns else pd.Series([""] * len(data))

    rows.extend(sample_rows(data, labels.eq("high_confidence_nearest_facade"), "strong_match"))
    rows.extend(sample_rows(data, labels.eq("needs_review_multiple_close_facades"), "ambiguous_match"))
    rows.extend(sample_rows(data, labels.eq("needs_review_no_building_match"), "no_building_match_failure", n=25))
    rows.extend(sample_rows(data, labels.eq("needs_review_multiple_close_facades") & evidence.str.contains("multiple_close_facades", na=False), "corner_or_close_facade_case"))

    tenant_counts = facade_ids[facade_ids.ne("")].value_counts()
    crowded_facades = set(tenant_counts[tenant_counts.gt(1)].index)
    rows.extend(sample_rows(data, facade_ids.isin(crowded_facades), "multi_tenant_or_shared_facade"))

    nonbuilding_summary = load_csv(NONBUILDING_ANALYSIS)
    nonbuilding_categories = set()
    if nonbuilding_summary is not None and "section" in nonbuilding_summary.columns:
        category_rows = nonbuilding_summary[nonbuilding_summary["section"].eq("likely_nonbuilding_category")]
        nonbuilding_categories = set(category_rows["group"].dropna().astype(str))
    if nonbuilding_categories and "poi_category" in data.columns:
        category_text = data["poi_category"].fillna("").astype(str).str.lower()
        basic_like_mask = pd.Series(False, index=data.index)
        for category in nonbuilding_categories:
            if category and category != "None":
                basic_like_mask |= category_text.str.contains(category.lower(), regex=False)
        rows.extend(sample_rows(data, basic_like_mask, "likely_nonbuilding_poi"))
    else:
        print("Could not derive non-building candidate rows from available columns.")

    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS).drop_duplicates(subset=["poi_id", "benchmark_type"])
    return candidates


def write_ground_truth_template(candidates):
    template = candidates[CANDIDATE_COLUMNS].copy()
    template["reviewer_label"] = ""
    template["reviewer_notes"] = ""
    template.to_csv(GROUND_TRUTH, index=False)
    print(f"Wrote ground-truth review template: {GROUND_TRUTH}")


def write_statistics(candidates):
    rows = []
    total = len(candidates)
    rows.append({"metric": "total_candidates", "group": "all", "count": total, "percent": 100.0 if total else 0.0})
    for column in ["benchmark_type", "region", "confidence_label", "ambiguity_bucket"]:
        if column not in candidates.columns:
            continue
        for value, count in candidates[column].fillna("").replace("", "(blank)").value_counts().items():
            rows.append(
                {
                    "metric": column,
                    "group": value,
                    "count": int(count),
                    "percent": round((int(count) / total) * 100, 2) if total else 0.0,
                }
            )
    ambiguous = candidates["confidence_label"].fillna("").astype(str).eq("needs_review_multiple_close_facades").sum()
    rows.append(
        {
            "metric": "ambiguity_rate",
            "group": "needs_review_multiple_close_facades",
            "count": int(ambiguous),
            "percent": round((int(ambiguous) / total) * 100, 2) if total else 0.0,
        }
    )
    pd.DataFrame(rows).to_csv(STATISTICS, index=False)
    print(f"Wrote benchmark statistics: {STATISTICS}")


def copy_review_packets():
    captions = load_csv(VIS_CAPTIONS)
    for folder in VIS_PACKET_MAP.values():
        (REVIEW_PACKETS / folder).mkdir(parents=True, exist_ok=True)
    if captions is None or captions.empty:
        print("No visualization captions found; review packet folders created without images.")
        return
    for _, row in captions.iterrows():
        source = Path(str(row.get("image_path", "")))
        group = str(row.get("group", ""))
        packet_group = VIS_PACKET_MAP.get(group)
        if not packet_group or not source.exists():
            continue
        target = REVIEW_PACKETS / packet_group / source.name
        shutil.copy2(source, target)
    print(f"Organized review packet images under: {REVIEW_PACKETS}")


def ensure_static_dirs():
    for directory in [
        BENCHMARK_DIR / "visualization_examples",
        BENCHMARK_DIR / "regions",
        REVIEW_PACKETS,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def run():
    ensure_static_dirs()
    candidates = build_candidates()
    candidates.to_csv(BENCHMARK_CANDIDATES, index=False)
    print(f"Wrote benchmark candidates: {BENCHMARK_CANDIDATES} ({len(candidates)} rows)")
    write_ground_truth_template(candidates)
    write_statistics(candidates)
    copy_review_packets()


if __name__ == "__main__":
    run()
