"""Final interpretable Problem 1 labels."""

from config import TEXT_AND_DISTANCE_FINAL_THRESHOLD
from normalization import is_missing, normalize_text

NON_BUILDING_POI_KEYWORDS = [
    "lake",
    "peak",
    "pass",
    "reservoir",
    "trail",
    "mountain",
    "park",
    "river",
    "creek",
    "campground",
    "viewpoint",
    "historic marker",
    "monument",
    "open space",
    "trailhead",
    "pond",
    "summit",
    "ridge",
    "falls",
    "waterfall",
    "natural",
    "wilderness",
    "recreation area",
    "overlook",
    "lookout",
    "cemetery",
    "tennis courts",
    "soccer",
    "historic district",
    "shelter",
    "flatirons",
    "courts",
    "field",
    "athletic field",
    "stadium",
    "sports complex",
    "plaza",
    "square",
]

BUSINESS_HINTS = [
    "restaurant",
    "cafe",
    "coffee",
    "shop",
    "store",
    "retail",
    "bank",
    "office",
    "clinic",
    "hotel",
    "bar",
    "salon",
    "pharmacy",
    "grocery",
    "business",
    "commercial",
]


def contains_keyword(text, keyword):
    """Match a normalized keyword as a phrase."""
    return f" {normalize_text(keyword)} " in f" {text} "


def classify_poi(row):
    """Classify whether a POI is expected to be building-address conflatable."""
    types_text = normalize_text(row.get("poi_types", ""))
    name_text = normalize_text(row.get("poi_name", ""))

    if any(contains_keyword(types_text, hint) for hint in BUSINESS_HINTS):
        return "building_expected_poi"
    if any(contains_keyword(types_text, keyword) for keyword in NON_BUILDING_POI_KEYWORDS):
        return "non_building_poi"
    if any(contains_keyword(name_text, keyword) for keyword in NON_BUILDING_POI_KEYWORDS):
        if not any(contains_keyword(name_text, hint) for hint in BUSINESS_HINTS):
            return "non_building_poi"
    return "building_expected_poi"


def assign_final_label(row):
    """Assign final Problem 1 labels from transparent evidence fields."""
    if row.get("poi_class") == "non_building_poi":
        return "non_building_poi"

    validation = str(row.get("building_validation_label", "")).lower()
    poi_address = row.get("poi_address_input")
    matched_address_id = row.get("matched_address_id")
    confidence = float(row.get("confidence") or 0)

    if validation == "high_confidence_strong":
        return "same_building_confirmed"
    if validation == "medium_confidence_valid":
        return "building_validated_candidate"
    if is_missing(poi_address) and not is_missing(matched_address_id):
        return "spatial_address_candidate"
    if is_missing(poi_address) and is_missing(matched_address_id):
        return "missing_poi_address_no_candidate"
    if not is_missing(poi_address) and confidence >= TEXT_AND_DISTANCE_FINAL_THRESHOLD:
        return "text_and_distance_match"
    return "needs_review"


def assign_match_method(row):
    """Derive the method used for the final match."""
    if row.get("final_label") == "non_building_poi":
        return "not_applicable"
    if is_missing(row.get("matched_address_id")):
        return "no_match"
    if is_missing(row.get("poi_address_input")):
        return "spatial_only"
    return "text_plus_distance"


def assign_review_reason(row):
    """Provide a concise review reason for each final label."""
    label = row.get("final_label")
    reasons = {
        "non_building_poi": "POI type/name suggests this entity does not require building or address conflation.",
        "same_building_confirmed": "POI and matched address are located in the same building polygon.",
        "building_validated_candidate": "POI/address match is supported by building validation but not the strongest building evidence.",
        "spatial_address_candidate": "POI has no address text, so match is based on nearest address and spatial evidence.",
        "missing_poi_address_no_candidate": "POI has no address text and no nearby address candidate was found.",
        "text_and_distance_match": "POI address text and distance both support the match.",
        "needs_review": "Match requires manual review due to weak, missing, or conflicting evidence.",
    }
    return reasons.get(label, reasons["needs_review"])


def enrich_labels(df):
    """Add POI class, final label, match method, and review reason."""
    result = df.copy()
    result["poi_class"] = result.apply(classify_poi, axis=1)
    result["final_label"] = result.apply(assign_final_label, axis=1)
    result["match_method"] = result.apply(assign_match_method, axis=1)
    result["review_reason"] = result.apply(assign_review_reason, axis=1)
    return result
