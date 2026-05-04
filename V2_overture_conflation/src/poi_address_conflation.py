# poi_address_conflation.py

import math
import pandas as pd
from difflib import SequenceMatcher
from shapely import wkb

# -----------------------------
# Config
# -----------------------------
POI_INPUT = r"..\outputs\candidate_fix_rules.csv"
ADDRESS_INPUT = r"..\data\overture_addresses.parquet"
OUTPUT_FILE = r"..\outputs\poi_address_matches.csv"

# Radius for nearby candidate addresses
MAX_CANDIDATE_DISTANCE_M = 100.0

# Confidence thresholds
HIGH_CONFIDENCE = 0.75
MEDIUM_CONFIDENCE = 0.45


# -----------------------------
# Utility functions
# -----------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    """
    Compute distance in meters between two lat/lon points.
    """
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return float("inf")

    r = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def normalize_text(value):
    """
    Lowercase and lightly normalize text for rough matching.
    """
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    replacements = {
        " street": " st",
        " avenue": " ave",
        " road": " rd",
        " boulevard": " blvd",
        " drive": " dr",
        " lane": " ln",
        " court": " ct",
        " place": " pl",
        ".": "",
        ",": "",
        "#": "",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return " ".join(value.split())


def similarity(a, b):
    """
    Sequence similarity from 0 to 1.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def distance_score(distance_m, max_distance=MAX_CANDIDATE_DISTANCE_M):
    """
    Convert distance into a 0..1 score.
    0m -> 1.0
    max_distance or beyond -> 0.0
    """
    if pd.isna(distance_m) or distance_m == float("inf"):
        return 0.0

    if distance_m >= max_distance:
        return 0.0

    return max(0.0, 1.0 - (distance_m / max_distance))


def parse_street_from_address(address):
    """
    Very rough extraction:
    '123 Main St' -> 'main st'
    """
    address = normalize_text(address)
    if not address:
        return ""

    parts = address.split()
    if not parts:
        return ""

    # Remove leading house number if present
    if parts[0].isdigit():
        parts = parts[1:]

    return " ".join(parts)

def extract_house_number(address):
    address = normalize_text(address)
    if not address:
        return ""
    parts = address.split()
    if parts and parts[0].isdigit():
        return parts[0]
    return ""

# -----------------------------
# Loading
# -----------------------------
def load_data():
    pois = pd.read_csv(POI_INPUT)
    pois = pois.head(1000)
    print(f"Using {len(pois)} POIs for testing")

    addresses = pd.read_parquet(ADDRESS_INPUT)
    addresses = addresses.sample(n=20000, random_state=42)
    print(f"Using {len(addresses)} address samples for testing")

    print("POI columns:")
    print(pois.columns.tolist())
    print("\nAddress columns:")
    print(addresses.columns.tolist())

    # Decode geometry bytes -> shapely point
    if "geometry" in addresses.columns:
        addresses["geometry_obj"] = addresses["geometry"].apply(
            lambda g: wkb.loads(g) if isinstance(g, (bytes, bytearray)) else g
        )

        addresses["longitude"] = addresses["geometry_obj"].apply(
            lambda g: g.x if g is not None else None
        )
        addresses["latitude"] = addresses["geometry_obj"].apply(
            lambda g: g.y if g is not None else None
        )

    # Build a freeform address string from Overture fields
    addresses["address"] = (
        addresses["number"].fillna("").astype(str).str.strip() + " " +
        addresses["street"].fillna("").astype(str).str.strip()
    ).str.strip()

    return pois, addresses


# -----------------------------
# Candidate generation
# -----------------------------
def find_candidates_for_poi(poi_row, addresses_df):
    """
    Return nearby address candidates within MAX_CANDIDATE_DISTANCE_M.
    Expects address CSV to have:
      - id
      - latitude
      - longitude
      - address or freeform
      - street
    """
    poi_lat = poi_row.get("entrance_lat")
    poi_lon = poi_row.get("entrance_lon")

    if pd.isna(poi_lat) or pd.isna(poi_lon):
        return pd.DataFrame()

    candidates = addresses_df.copy()

    candidates["distance_m"] = candidates.apply(
        lambda row: haversine_m(
            poi_lat,
            poi_lon,
            row.get("latitude"),
            row.get("longitude"),
        ),
        axis=1,
    )

    candidates = candidates[candidates["distance_m"] <= MAX_CANDIDATE_DISTANCE_M].copy()
    return candidates.sort_values("distance_m")


# -----------------------------
# Scoring
# -----------------------------
def score_candidate(poi_row, addr_row):
    """
    Combine:
      - distance
      - full address similarity
      - street similarity
    """
    poi_address = poi_row.get("address", "")
    poi_street = poi_row.get("entrance_street", "")
    if not poi_street:
        poi_street = parse_street_from_address(poi_address)

    addr_full = addr_row.get("address", "")
    if not addr_full:
        addr_full = addr_row.get("freeform", "")

    addr_street = addr_row.get("street", "")
    if not addr_street:
        addr_street = parse_street_from_address(addr_full)

    dist = addr_row.get("distance_m", float("inf"))
    dist_s = distance_score(dist)
    addr_s = similarity(poi_address, addr_full)
    street_s = similarity(poi_street, addr_street)

    poi_number = extract_house_number(poi_address)
    addr_number = str(addr_row.get("number", "")).strip()

    number_score = 0.0
    if poi_number and addr_number:
        if poi_number == addr_number:
            number_score = 1.0
        else:
            number_score = -0.5

    # Weighted baseline
    score = (
    0.40 * dist_s
    + 0.25 * addr_s
    + 0.15 * street_s
    + 0.20 * max(0.0, number_score)
)

    if number_score < 0:
        score -= 0.20

    return {
        "distance_score": dist_s,
        "address_similarity": addr_s,
        "street_similarity": street_s,
        "confidence": score,
    }


def classify_match(confidence, candidate_count):
    if candidate_count == 0:
        return "no_candidate"
    if confidence >= HIGH_CONFIDENCE:
        return "matched_high"
    if confidence >= MEDIUM_CONFIDENCE:
        return "matched_medium"
    return "uncertain"


# -----------------------------
# Main matching logic
# -----------------------------
def match_pois_to_addresses(pois_df, addresses_df):
    results = []

    for i, (_, poi) in enumerate(pois_df.iterrows()):
        if i % 100 == 0:
            print(f"Processing POI {i}")
        poi_id = poi.get("id")
        poi_name = poi.get("name", "")
        poi_address = poi.get("address", "")

        candidates = find_candidates_for_poi(poi, addresses_df)

        if len(candidates) == 0:
            results.append({
                "poi_id": poi_id,
                "poi_name": poi_name,
                "poi_address_input": poi_address,
                "best_address_id": None,
                "best_address_text": None,
                "distance_m": None,
                "distance_score": 0.0,
                "address_similarity": 0.0,
                "street_similarity": 0.0,
                "confidence": 0.0,
                "candidate_count": 0,
                "status": "no_candidate",
            })
            continue

                # Store every candidate address along with its score
        scored_rows = []
        for _, addr in candidates.iterrows():
            scores = score_candidate(poi, addr)

            # Build readable full address text
            addr_full = addr.get("address", "")
            if not addr_full:
                addr_full = addr.get("freeform", "")

            scored_rows.append({
                "address_id": addr.get("id"),
                "address_text": addr_full,

                # Save candidate address coordinates
                # We need these later for real building validation
                "address_latitude": addr.get("latitude"),
                "address_longitude": addr.get("longitude"),

                "distance_m": addr.get("distance_m"),
                **scores,
            })

        scored_df = pd.DataFrame(scored_rows).sort_values(
            by=["confidence", "distance_m"],
            ascending=[False, True]
        )

        best = scored_df.iloc[0]
        status = classify_match(best["confidence"], len(scored_df))

                # Save the best matched address for this POI
        results.append({
            "poi_id": poi_id,
            "poi_name": poi_name,
            "poi_address_input": poi_address,

            "best_address_id": best["address_id"],
            "best_address_text": best["address_text"],

            # Save matched address coordinates for later building checks
            "best_address_lat": best["address_latitude"],
            "best_address_lon": best["address_longitude"],

            "distance_m": best["distance_m"],
            "distance_score": best["distance_score"],
            "address_similarity": best["address_similarity"],
            "street_similarity": best["street_similarity"],
            "confidence": best["confidence"],
            "candidate_count": len(scored_df),
            "status": status,
        })

    return pd.DataFrame(results)


# -----------------------------
# Entry point
# -----------------------------
def main():
    pois_df, addresses_df = load_data()

    # Optional cleanup: drop rows without entrance coordinates
    pois_df = pois_df.dropna(subset=["entrance_lat", "entrance_lon"]).copy()

    results_df = match_pois_to_addresses(pois_df, addresses_df)
    results_df.to_csv(OUTPUT_FILE, index=False)
    # Split results for easier inspection
    results_df[results_df["status"] == "matched_medium"].to_csv(
        r"..\outputs\poi_address_matched_medium.csv", index=False
    )

    results_df[results_df["status"] == "uncertain"].to_csv(
        r"..\outputs\poi_address_uncertain.csv", index=False
    )

    results_df[results_df["status"] == "no_candidate"].to_csv(
        r"..\outputs\poi_address_no_candidate.csv", index=False
    )


    print(f"Saved {len(results_df)} matches to {OUTPUT_FILE}")
    print(results_df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()