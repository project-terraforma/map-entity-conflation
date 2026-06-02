"""Evaluate Gauri Problem 1 quality and Problem 2 proxy agreement.

Problem 1 has no verified golden dataset yet, so its automated score measures
internal consistency and coverage. Human-reviewed rows in the generated review
queue are reported separately as pilot accuracy metrics.

Problem 2 uses the checked-in facade proxy benchmark. Those metrics measure
agreement with sign and entrance proxy points, not human-verified accuracy.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVALUATION_ROOT.parent
OUTPUT_DIR = EVALUATION_ROOT / "outputs"

DEFAULT_P1_INPUT = REPO_ROOT / "Conflation_pipeline" / "outputs" / "final_problem1_conflation.csv"
DEFAULT_P2_INPUT = (
    REPO_ROOT
    / "Problem2_facade_pipeline"
    / "benchmark_dataset"
    / "problem2_proxy_evaluated_building_based_rows.csv"
)

SUMMARY_OUTPUT = OUTPUT_DIR / "evaluation_summary.csv"
P1_METRICS_OUTPUT = OUTPUT_DIR / "problem1_quality_metrics.csv"
P1_FLAGS_OUTPUT = OUTPUT_DIR / "problem1_automated_flags.csv"
P1_REVIEW_QUEUE_OUTPUT = OUTPUT_DIR / "problem1_review_queue.csv"
P1_RANDOM_REVIEW_QUEUE_OUTPUT = OUTPUT_DIR / "problem1_random_review_queue.csv"
P1_REVIEWED_METRICS_OUTPUT = OUTPUT_DIR / "problem1_reviewed_metrics.csv"
P1_LLM_METRICS_OUTPUT = OUTPUT_DIR / "problem1_llm_silver_metrics.csv"
P1_LLM_REVIEW_OUTPUT = OUTPUT_DIR / "problem1_llm_silver_review.csv"
P2_METRICS_OUTPUT = OUTPUT_DIR / "problem2_proxy_metrics.csv"

REVIEW_COLUMNS = [
    "review_address_verdict",
    "review_building_verdict",
    "review_street_verdict",
    "review_poi_class_verdict",
    "review_overall_verdict",
    "evidence_source_urls",
    "review_notes",
    "reviewed_by",
    "reviewed_at",
]

P1_REVIEW_COLUMNS = [
    "poi_id",
    "poi_name",
    "poi_lat",
    "poi_lon",
    "poi_address_input",
    "poi_types",
    "matched_address_id",
    "matched_address_text",
    "matched_building_id",
    "matched_street",
    "nearest_street_name",
    "street_segment_match_status",
    "distance_m",
    "confidence",
    "candidate_count",
    "address_match_status",
    "building_status",
    "real_building_relation",
    "building_validation_label",
    "poi_class",
    "final_label",
    "review_reason",
    "evaluation_bucket",
    "automated_quality_score",
    "automated_flags",
]

TRUE_VALUES = {"1", "true", "yes", "y"}
REVIEWED_VALUES = {"correct", "incorrect", "ambiguous", "uncertain", "not_applicable"}
LLM_VERDICT_COLUMNS = [
    ("llm_address_verdict", "llm_silver_address_accuracy"),
    ("llm_building_verdict", "llm_silver_building_accuracy"),
    ("llm_street_verdict", "llm_silver_street_accuracy"),
    ("llm_poi_class_verdict", "llm_silver_poi_class_accuracy"),
    ("llm_overall_verdict", "llm_silver_full_chain_accuracy"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as string dictionaries."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    """Write dictionaries to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def text(value: object) -> str:
    """Normalize a CSV value for comparisons."""
    return str(value or "").strip()


def number(value: object) -> float | None:
    """Parse a CSV number when possible."""
    try:
        return float(text(value))
    except ValueError:
        return None


def is_true(value: object) -> bool:
    """Parse common CSV boolean values."""
    return text(value).lower() in TRUE_VALUES


def percent(numerator: int | float, denominator: int | float) -> float:
    """Return a percentage rounded for reports."""
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def metric(
    metric_name: str,
    value: int | float | str,
    unit: str,
    description: str,
    numerator: int | float | str = "",
    denominator: int | float | str = "",
) -> dict[str, object]:
    """Create one report metric row."""
    return {
        "metric": metric_name,
        "value": value,
        "unit": unit,
        "numerator": numerator,
        "denominator": denominator,
        "description": description,
    }


def p1_flags(row: dict[str, str]) -> list[str]:
    """Return automated Problem 1 review flags for one POI."""
    flags = []
    poi_class = text(row.get("poi_class"))
    if poi_class != "non_building_poi" and not text(row.get("matched_address_id")):
        flags.append("missing_address_match")
    if poi_class != "non_building_poi" and not text(row.get("matched_building_id")):
        flags.append("missing_building_match")
    if text(row.get("building_status")) == "building_conflict":
        flags.append("building_conflict")
    if text(row.get("real_building_relation")) == "different_building":
        flags.append("different_poi_and_address_buildings")
    distance = number(row.get("distance_m"))
    if distance is not None and distance > 75.0:
        flags.append("large_poi_address_distance")
    candidates = number(row.get("candidate_count"))
    if candidates is not None and candidates >= 100:
        flags.append("many_address_candidates")
    if (
        text(row.get("nearest_street_name"))
        and text(row.get("matched_street"))
        and text(row.get("street_segment_match_status")) == "street_segment_nearest_only"
    ):
        flags.append("named_street_segment_mismatch")
    if text(row.get("final_label")) == "needs_review":
        flags.append("pipeline_needs_review")
    return flags


def p1_quality_score(row: dict[str, str]) -> int:
    """Return an explainable 0..100 internal-quality score for one POI."""
    if text(row.get("poi_class")) == "non_building_poi":
        return 100 if text(row.get("final_label")) == "non_building_poi" else 0

    score = 0
    if text(row.get("matched_address_id")):
        score += 20
    if text(row.get("matched_building_id")):
        score += 20
    if text(row.get("real_building_relation")) == "same_building":
        score += 30
    if text(row.get("address_match_status")) in {"matched_high", "matched_medium"}:
        score += 15
    if text(row.get("matched_street")):
        score += 10
    if text(row.get("street_segment_match_status")) == "street_segment_name_match":
        score += 5
    return score


def build_p1_metrics(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build Problem 1 quality metrics and row-level automated flags."""
    if not rows:
        return [], []

    relevant = [row for row in rows if text(row.get("poi_class")) != "non_building_poi"]
    scores = [p1_quality_score(row) for row in rows]
    relevant_scores = [p1_quality_score(row) for row in relevant]
    address_matches = sum(bool(text(row.get("matched_address_id"))) for row in relevant)
    building_matches = sum(bool(text(row.get("matched_building_id"))) for row in relevant)
    same_building = sum(text(row.get("real_building_relation")) == "same_building" for row in relevant)
    streets = sum(bool(text(row.get("matched_street"))) for row in relevant)
    named_segments = sum(bool(text(row.get("nearest_street_name"))) for row in relevant)
    segment_name_matches = sum(
        text(row.get("street_segment_match_status")) == "street_segment_name_match" for row in relevant
    )
    needs_review = sum(text(row.get("final_label")) == "needs_review" for row in rows)
    non_building = sum(text(row.get("poi_class")) == "non_building_poi" for row in rows)

    metrics = [
        metric("total_pois", len(rows), "rows", "All Problem 1 output rows."),
        metric("building_relevant_pois", len(relevant), "rows", "Rows expected to relate to a building."),
        metric(
            "internal_quality_score_all_pois",
            round(sum(scores) / len(scores), 2),
            "score_out_of_100",
            "Average automated evidence score. This is not verified accuracy.",
        ),
        metric(
            "internal_quality_score_building_relevant",
            round(sum(relevant_scores) / len(relevant_scores), 2) if relevant_scores else 0.0,
            "score_out_of_100",
            "Average automated evidence score for building-relevant POIs. This is not verified accuracy.",
        ),
        metric("matched_address_rate", percent(address_matches, len(relevant)), "percent", "Building-relevant POIs with an address match.", address_matches, len(relevant)),
        metric("matched_building_rate", percent(building_matches, len(relevant)), "percent", "Building-relevant POIs with a building match.", building_matches, len(relevant)),
        metric("same_building_support_rate", percent(same_building, len(relevant)), "percent", "Building-relevant POIs whose POI and matched address support the same building.", same_building, len(relevant)),
        metric("street_extraction_rate", percent(streets, len(relevant)), "percent", "Building-relevant POIs with a street extracted from the matched address.", streets, len(relevant)),
        metric("named_street_segment_availability_rate", percent(named_segments, len(relevant)), "percent", "Building-relevant POIs whose nearest street segment contains a usable name.", named_segments, len(relevant)),
        metric("street_segment_name_agreement_rate", percent(segment_name_matches, named_segments), "percent", "Named nearest street segments whose name agrees with the matched street.", segment_name_matches, named_segments),
        metric("needs_review_rate", percent(needs_review, len(rows)), "percent", "Rows explicitly sent to manual review by Problem 1.", needs_review, len(rows)),
        metric("non_building_poi_rate", percent(non_building, len(rows)), "percent", "Rows classified as not requiring building storefront conflation.", non_building, len(rows)),
    ]

    flagged_rows = []
    for row in rows:
        flags = p1_flags(row)
        if flags:
            flagged_rows.append(
                {
                    "poi_id": row.get("poi_id", ""),
                    "poi_name": row.get("poi_name", ""),
                    "final_label": row.get("final_label", ""),
                    "building_status": row.get("building_status", ""),
                    "automated_quality_score": p1_quality_score(row),
                    "automated_flags": "|".join(flags),
                }
            )
    return metrics, flagged_rows


def p1_bucket(row: dict[str, str]) -> str:
    """Assign one sampling bucket to a Problem 1 row."""
    flags = set(p1_flags(row))
    if "building_conflict" in flags or "different_poi_and_address_buildings" in flags:
        return "building_conflict"
    if text(row.get("poi_class")) == "non_building_poi":
        return "non_building_poi"
    if text(row.get("final_label")) == "needs_review":
        return "needs_review"
    if text(row.get("final_label")) == "text_and_distance_match":
        return "text_and_distance_match"
    if text(row.get("final_label")) == "building_validated_candidate":
        return "building_validated_candidate"
    if text(row.get("final_label")) == "same_building_confirmed":
        return "same_building_confirmed"
    return "other"


def existing_review_lookup(path: Path) -> dict[str, dict[str, str]]:
    """Keep completed human reviews when regenerating a queue."""
    return {text(row.get("poi_id")): row for row in read_csv(path) if text(row.get("poi_id"))}


def format_review_queue(
    selected: list[dict[str, str]],
    previous_path: Path,
    queue_type: str,
) -> list[dict[str, object]]:
    """Format sampled Problem 1 rows and preserve existing human reviews."""
    previous = existing_review_lookup(previous_path)
    output = []
    for row in selected:
        poi_id = text(row.get("poi_id"))
        item = {column: row.get(column, "") for column in P1_REVIEW_COLUMNS}
        item["evaluation_bucket"] = p1_bucket(row)
        item["automated_quality_score"] = p1_quality_score(row)
        item["automated_flags"] = "|".join(p1_flags(row))
        item["queue_type"] = queue_type
        for column in REVIEW_COLUMNS:
            item[column] = previous.get(poi_id, {}).get(column, "")
        output.append(item)
    return output


def build_review_queue(rows: list[dict[str, str]], size: int, seed: int) -> list[dict[str, object]]:
    """Create a balanced Problem 1 human-review queue."""
    if not rows or size <= 0:
        return []

    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[p1_bucket(row)].append(row)
    for bucket_rows in grouped.values():
        rng.shuffle(bucket_rows)

    selected = []
    seen = set()
    bucket_order = [
        "same_building_confirmed",
        "text_and_distance_match",
        "needs_review",
        "building_conflict",
        "non_building_poi",
        "building_validated_candidate",
        "other",
    ]
    per_bucket = max(1, size // max(1, len(bucket_order)))
    for bucket in bucket_order:
        for row in grouped.get(bucket, [])[:per_bucket]:
            poi_id = text(row.get("poi_id"))
            if poi_id and poi_id not in seen:
                selected.append(row)
                seen.add(poi_id)

    remaining = list(rows)
    rng.shuffle(remaining)
    for row in remaining:
        if len(selected) >= min(size, len(rows)):
            break
        poi_id = text(row.get("poi_id"))
        if poi_id and poi_id not in seen:
            selected.append(row)
            seen.add(poi_id)

    return format_review_queue(selected, P1_REVIEW_QUEUE_OUTPUT, "balanced_challenge_sample")


def build_random_review_queue(rows: list[dict[str, str]], size: int, seed: int) -> list[dict[str, object]]:
    """Create a reproducible simple-random Problem 1 review queue."""
    if not rows or size <= 0:
        return []
    selected = random.Random(seed).sample(rows, k=min(size, len(rows)))
    return format_review_queue(selected, P1_RANDOM_REVIEW_QUEUE_OUTPUT, "simple_random_sample")


def build_reviewed_metrics(review_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Report optional human-reviewed Problem 1 accuracy metrics."""
    metrics = []
    for column, label in [
        ("review_address_verdict", "human_reviewed_address_accuracy"),
        ("review_building_verdict", "human_reviewed_building_accuracy"),
        ("review_street_verdict", "human_reviewed_street_accuracy"),
        ("review_poi_class_verdict", "human_reviewed_poi_class_accuracy"),
        ("review_overall_verdict", "human_reviewed_full_chain_accuracy"),
    ]:
        eligible = [row for row in review_rows if text(row.get(column)).lower() in REVIEWED_VALUES]
        decided = [row for row in eligible if text(row.get(column)).lower() in {"correct", "incorrect"}]
        correct = sum(text(row.get(column)).lower() == "correct" for row in decided)
        metrics.append(
            metric(
                label,
                percent(correct, len(decided)),
                "percent",
                "Human-reviewed pilot accuracy. Ambiguous, uncertain, and not-applicable rows are excluded from the denominator.",
                correct,
                len(decided),
            )
        )
        metrics.append(
            metric(
                f"{label}_reviewed_rows",
                len(eligible),
                "rows",
                "Rows with any completed human-review verdict, including ambiguous outcomes.",
            )
        )
    return metrics


def build_llm_silver_metrics(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Report LLM-assisted silver metrics separately from golden-set metrics."""
    if not rows:
        return []

    metrics = [
        metric(
            "llm_silver_rows",
            len(rows),
            "rows",
            "LLM-researched rows. These are silver labels and require human approval before being treated as accuracy.",
        )
    ]
    for column, label in LLM_VERDICT_COLUMNS:
        eligible = [row for row in rows if text(row.get(column)).lower() in REVIEWED_VALUES]
        decided = [row for row in eligible if text(row.get(column)).lower() in {"correct", "incorrect"}]
        correct = sum(text(row.get(column)).lower() == "correct" for row in decided)
        metrics.append(
            metric(
                label,
                percent(correct, len(decided)),
                "percent",
                "LLM-assisted silver estimate. Ambiguous, uncertain, and not-applicable rows are excluded. This is not human-reviewed accuracy.",
                correct,
                len(decided),
            )
        )
        metrics.append(
            metric(
                f"{label}_decided_rows",
                len(decided),
                "rows",
                "LLM silver rows labeled correct or incorrect for this field.",
            )
        )

    confidence_counts = Counter(text(row.get("llm_confidence")).lower() or "missing" for row in rows)
    for confidence in ["high", "medium", "low", "missing"]:
        count = confidence_counts.get(confidence, 0)
        if count:
            metrics.append(
                metric(
                    f"llm_confidence_{confidence}_rate",
                    percent(count, len(rows)),
                    "percent",
                    "Share of LLM-researched rows with this confidence label.",
                    count,
                    len(rows),
                )
            )
    evidence_rows = sum(bool(text(row.get("llm_evidence_urls") or row.get("evidence_source_urls"))) for row in rows)
    metrics.append(
        metric(
            "llm_rows_with_evidence_urls_rate",
            percent(evidence_rows, len(rows)),
            "percent",
            "LLM-researched rows containing one or more evidence URLs.",
            evidence_rows,
            len(rows),
        )
    )
    return metrics


def build_p2_metrics(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Build Problem 2 facade proxy-agreement metrics."""
    if not rows:
        return []
    strict = sum(is_true(row.get("proxy_agreement")) for row in rows)
    baseline = sum(is_true(row.get("nearest_edge_proxy_agreement")) for row in rows)
    corner_tolerant = sum(is_true(row.get("near_correct_under_corner_rule")) for row in rows)
    corner_ambiguous = sum(is_true(row.get("is_corner_ambiguous")) for row in rows)
    shared = sum(is_true(row.get("is_shared_building")) for row in rows)
    return [
        metric("evaluated_proxy_rows", len(rows), "rows", "Building-based sign or entrance proxy observations evaluated."),
        metric("strict_facade_proxy_agreement", percent(strict, len(rows)), "percent", "Selected facade exactly equals the facade nearest to the sign or entrance proxy point. This is not human-verified accuracy.", strict, len(rows)),
        metric("nearest_edge_baseline_proxy_agreement", percent(baseline, len(rows)), "percent", "Simple nearest-edge baseline agreement with the proxy facade.", baseline, len(rows)),
        metric("reranker_improvement_over_baseline", round(percent(strict, len(rows)) - percent(baseline, len(rows)), 2), "percentage_points", "Strict proxy-agreement improvement over the nearest-edge baseline."),
        metric("corner_tolerant_proxy_agreement", percent(corner_tolerant, len(rows)), "percent", "Diagnostic agreement allowing plausible adjacent facade edges near corners.", corner_tolerant, len(rows)),
        metric("corner_ambiguous_rate", percent(corner_ambiguous, len(rows)), "percent", "Proxy observations where two facade edges are close alternatives.", corner_ambiguous, len(rows)),
        metric("shared_building_rate", percent(shared, len(rows)), "percent", "Evaluated proxy rows belonging to buildings shared by multiple POIs.", shared, len(rows)),
    ]


def summary_rows(
    p1_metrics: list[dict[str, object]],
    reviewed_metrics: list[dict[str, object]],
    llm_silver_metrics: list[dict[str, object]],
    p2_metrics: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return a concise top-level dashboard."""
    combined = {row["metric"]: row for row in p1_metrics + reviewed_metrics + llm_silver_metrics + p2_metrics}
    wanted = [
        ("problem1", "internal_quality_score_building_relevant"),
        ("problem1", "matched_address_rate"),
        ("problem1", "matched_building_rate"),
        ("problem1", "same_building_support_rate"),
        ("problem1", "needs_review_rate"),
        ("problem1_human_review", "human_reviewed_full_chain_accuracy"),
        ("problem1_llm_silver", "llm_silver_full_chain_accuracy"),
        ("problem1_llm_silver", "llm_silver_address_accuracy"),
        ("problem1_llm_silver", "llm_silver_street_accuracy"),
        ("problem2", "strict_facade_proxy_agreement"),
        ("problem2", "nearest_edge_baseline_proxy_agreement"),
        ("problem2", "reranker_improvement_over_baseline"),
        ("problem2", "corner_tolerant_proxy_agreement"),
    ]
    rows = []
    for problem, name in wanted:
        if name in combined:
            rows.append({"problem": problem, **combined[name]})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-input", type=Path, default=DEFAULT_P1_INPUT)
    parser.add_argument("--p2-input", type=Path, default=DEFAULT_P2_INPUT)
    parser.add_argument(
        "--llm-silver-input",
        type=Path,
        help="Optional LLM-researched Problem 1 review CSV. Reported separately as silver-label metrics.",
    )
    parser.add_argument("--review-sample-size", type=int, default=50)
    parser.add_argument("--random-review-sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    p1_rows = read_csv(args.p1_input)
    p2_rows = read_csv(args.p2_input)
    llm_silver_rows = read_csv(args.llm_silver_input) if args.llm_silver_input else []
    if not p1_rows:
        raise SystemExit(f"Problem 1 input not found or empty: {args.p1_input}")

    p1_metrics, flagged_rows = build_p1_metrics(p1_rows)
    review_queue = build_review_queue(p1_rows, args.review_sample_size, args.seed)
    random_review_queue = build_random_review_queue(p1_rows, args.random_review_sample_size, args.seed)
    reviewed_metrics = build_reviewed_metrics(review_queue)
    llm_silver_metrics = build_llm_silver_metrics(llm_silver_rows)
    p2_metrics = build_p2_metrics(p2_rows)

    write_csv(P1_METRICS_OUTPUT, p1_metrics)
    write_csv(P1_FLAGS_OUTPUT, flagged_rows)
    review_fieldnames = P1_REVIEW_COLUMNS + ["queue_type"] + REVIEW_COLUMNS
    write_csv(P1_REVIEW_QUEUE_OUTPUT, review_queue, review_fieldnames)
    write_csv(P1_RANDOM_REVIEW_QUEUE_OUTPUT, random_review_queue, review_fieldnames)
    write_csv(P1_REVIEWED_METRICS_OUTPUT, reviewed_metrics)
    write_csv(P1_LLM_METRICS_OUTPUT, llm_silver_metrics)
    if llm_silver_rows:
        write_csv(P1_LLM_REVIEW_OUTPUT, llm_silver_rows)
    write_csv(P2_METRICS_OUTPUT, p2_metrics)
    write_csv(SUMMARY_OUTPUT, summary_rows(p1_metrics, reviewed_metrics, llm_silver_metrics, p2_metrics))

    print(f"Problem 1 rows evaluated: {len(p1_rows)}")
    print(f"Problem 1 flagged rows: {len(flagged_rows)}")
    print(f"Problem 1 review queue rows: {len(review_queue)}")
    print(f"Problem 1 random review queue rows: {len(random_review_queue)}")
    print(f"Problem 1 LLM silver rows: {len(llm_silver_rows)}")
    print(f"Problem 2 proxy rows evaluated: {len(p2_rows)}")
    print(f"Wrote evaluation dashboard: {SUMMARY_OUTPUT}")
    print("Problem 1 automated scores are internal-quality metrics, not verified accuracy.")
    print("Problem 2 automated scores are facade proxy-agreement metrics, not human-verified accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
