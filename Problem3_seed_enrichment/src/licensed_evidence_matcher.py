"""Match seeds to an explicitly licensed external POI evidence feed."""

from pathlib import Path

import pandas as pd

from config import EXTERNAL_AMBIGUITY_MARGIN, EXTERNAL_SEARCH_RADIUS_M, EXTERNAL_VERIFIED_NAME_SCORE, EXTERNAL_VERIFIED_SCORE
from local_overture_matcher import haversine_m, score_match
from text_utils import clean_text

REQUIRED_COLUMNS = {"name", "lat", "lon", "source_url", "license"}
OPTIONAL_COLUMNS = {
    "provider": "",
    "provider_id": "",
    "address": "",
    "category": "",
    "business_status": "",
    "website": "",
    "phone": "",
}


def load_licensed_evidence(path):
    """Load a user-supplied feed whose terms allow downstream contribution."""
    evidence_path = Path(path)
    if not evidence_path.exists():
        raise FileNotFoundError(f"Licensed evidence CSV does not exist: {evidence_path}")
    df = pd.read_csv(evidence_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{evidence_path} is missing required columns: {sorted(missing)}")
    for column, default in OPTIONAL_COLUMNS.items():
        if column not in df.columns:
            df[column] = default
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df[df["lat"].notna() & df["lon"].notna()].copy()


def _empty_match(status="not_found"):
    return {"external_match_status": status, "external_score": 0.0}


def match_one(seed, evidence, radius_m=EXTERNAL_SEARCH_RADIUS_M):
    """Return the best licensed evidence candidate for one seed."""
    candidates = []
    for _, candidate in evidence.iterrows():
        distance_m = haversine_m(seed["seed_lat"], seed["seed_lon"], candidate["lat"], candidate["lon"])
        if distance_m > radius_m:
            continue
        score, name_score = score_match(seed["poi_name_seed"], candidate["name"], distance_m, radius_m)
        candidates.append((score, distance_m, name_score, candidate))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates:
        return _empty_match()
    score, distance_m, name_score, candidate = candidates[0]
    runner_up_score = candidates[1][0] if len(candidates) > 1 else 0.0
    ambiguous = score - runner_up_score < EXTERNAL_AMBIGUITY_MARGIN
    status = "verified" if score >= EXTERNAL_VERIFIED_SCORE and name_score >= EXTERNAL_VERIFIED_NAME_SCORE and not ambiguous else "needs_review"
    return {
        "external_match_status": status,
        "external_score": score,
        "external_name_score": name_score,
        "external_distance_m": round(distance_m, 3),
        "external_runner_up_score": runner_up_score,
        "external_candidate_count": len(candidates),
        "external_provider": clean_text(candidate.get("provider")),
        "external_provider_id": clean_text(candidate.get("provider_id")),
        "external_name": clean_text(candidate.get("name")),
        "external_address": clean_text(candidate.get("address")),
        "external_lat": candidate.get("lat", ""),
        "external_lon": candidate.get("lon", ""),
        "external_category": clean_text(candidate.get("category")),
        "external_business_status": clean_text(candidate.get("business_status")),
        "external_website": clean_text(candidate.get("website")),
        "external_phone": clean_text(candidate.get("phone")),
        "external_source_url": clean_text(candidate.get("source_url")),
        "external_license": clean_text(candidate.get("license")),
    }


def match_seeds(seeds, evidence):
    """Attach licensed external evidence to unresolved imagery seeds."""
    rows = []
    for _, row in seeds.iterrows():
        if row.get("overture_id_snapshot") or row.get("local_overture_match_status") == "confirmed":
            rows.append({"external_match_status": "skipped_local_confirmation"})
        else:
            rows.append(match_one(row, evidence))
    return pd.concat([seeds.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

