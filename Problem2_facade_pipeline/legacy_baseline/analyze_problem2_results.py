"""Analyze Problem 2 facade matching outputs without changing match logic."""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MATCHES_CSV = OUTPUT_DIR / "problem2_facade_matches.csv"
REVIEW_SAMPLE_CSV = OUTPUT_DIR / "manual_review_facade_sample.csv"
SUMMARY_CSV = OUTPUT_DIR / "problem2_summary.csv"
ANALYSIS_CSV = OUTPUT_DIR / "problem2_result_analysis.csv"


def load_csv_if_available(path):
    """Load a CSV if it exists, otherwise return None and print a note."""
    if not path.exists():
        print(f"Optional file not found: {path}")
        return None
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {path}: {len(df)} rows")
    return df


def has_column(df, column):
    """Return whether a dataframe has a column, printing a note if absent."""
    if column in df.columns:
        return True
    print(f"Column not found, skipping: {column}")
    return False


def nonempty_count(series):
    """Count values that are not null and not blank after string conversion."""
    return int(series.fillna("").astype(str).str.strip().ne("").sum())


def add_metric(rows, metric, value, detail=""):
    rows.append({"metric": metric, "value": value, "detail": detail})


def add_bucket(rows, name, count, total, detail=""):
    percent = round((count / total) * 100, 2) if total else 0.0
    rows.append({"metric": "bucket", "value": count, "detail": f"{name}; percent={percent}; {detail}".rstrip("; ")})
    print(f"{name}: {count} ({percent}%)")


def print_confidence_counts(df, rows):
    if not has_column(df, "confidence_label"):
        return
    total = len(df)
    counts = df["confidence_label"].fillna("").replace("", "(blank)").value_counts(dropna=False)
    print("\nConfidence label counts:")
    for label, count in counts.items():
        percent = round((count / total) * 100, 2) if total else 0.0
        print(f"{label}: {count} ({percent}%)")
        rows.append({"metric": "confidence_label", "value": int(count), "detail": f"{label}; percent={percent}"})


def print_evidence_patterns(df, rows, max_patterns=15):
    if not has_column(df, "evidence_fields_used"):
        return
    evidence = df["evidence_fields_used"].fillna("").astype(str).str.strip()
    evidence = evidence[evidence.ne("")]
    if evidence.empty:
        print("\nEvidence column exists but has no non-empty values.")
        add_metric(rows, "evidence_patterns_nonempty", 0)
        return

    counts = evidence.value_counts().head(max_patterns)
    print("\nTop evidence patterns:")
    for pattern, count in counts.items():
        print(f"{count}: {pattern}")
        rows.append({"metric": "evidence_pattern", "value": int(count), "detail": pattern})


def identify_buckets(df, rows):
    total = len(df)
    print("\nFailure/review buckets:")

    labels = None
    if has_column(df, "confidence_label"):
        labels = df["confidence_label"].fillna("").astype(str)
        add_bucket(rows, "no building match", int(labels.eq("needs_review_no_building_match").sum()), total)
        add_bucket(rows, "multiple close facades", int(labels.eq("needs_review_multiple_close_facades").sum()), total)
        add_bucket(rows, "street-supported matches", int(labels.str.contains("street_supported", na=False).sum()), total)
        add_bucket(
            rows,
            "inside-building nearest-edge matches",
            int(labels.eq("medium_confidence_inside_building_nearest_edge").sum()),
            total,
        )
    else:
        print("Cannot identify label-based buckets because confidence_label is missing.")

    if has_column(df, "method_used"):
        methods = df["method_used"].fillna("").astype(str)
        nearest_only = methods.eq("nearest_facade_edge")
        if labels is not None:
            nearest_only &= ~labels.str.contains("street_supported", na=False)
        if "evidence_fields_used" in df.columns:
            evidence = df["evidence_fields_used"].fillna("").astype(str)
            nearest_only &= ~evidence.str.contains("street_distance_m=", regex=False, na=False)
        add_bucket(rows, "nearest-facade-only matches", int(nearest_only.sum()), total)
    else:
        print("Cannot identify method-based nearest-facade-only bucket because method_used is missing.")


def summarize_optional_files(rows):
    review_df = load_csv_if_available(REVIEW_SAMPLE_CSV)
    if review_df is not None:
        add_metric(rows, "manual_review_sample_rows", len(review_df), str(REVIEW_SAMPLE_CSV))
        print(f"Manual review sample columns: {list(review_df.columns)}")

    summary_df = load_csv_if_available(SUMMARY_CSV)
    if summary_df is not None:
        add_metric(rows, "problem2_summary_rows", len(summary_df), str(SUMMARY_CSV))
        print(f"Problem 2 summary columns: {list(summary_df.columns)}")


def run():
    if not MATCHES_CSV.exists():
        raise FileNotFoundError(f"Required facade match results not found: {MATCHES_CSV}")

    df = pd.read_csv(MATCHES_CSV, low_memory=False)
    rows = []

    print(f"Loaded facade matches: {MATCHES_CSV}")
    print(f"Columns: {list(df.columns)}")

    total = len(df)
    print(f"\nTotal POIs processed: {total}")
    add_metric(rows, "total_pois_processed", total)

    if has_column(df, "facade_id"):
        facade_count = nonempty_count(df["facade_id"])
        percent = round((facade_count / total) * 100, 2) if total else 0.0
        print(f"POIs with facade IDs: {facade_count} ({percent}%)")
        add_metric(rows, "pois_with_facade_ids", facade_count, f"percent={percent}")

    print_confidence_counts(df, rows)
    print_evidence_patterns(df, rows)
    identify_buckets(df, rows)
    summarize_optional_files(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ANALYSIS_CSV, index=False)
    print(f"\nWrote analysis CSV: {ANALYSIS_CSV}")


if __name__ == "__main__":
    run()
