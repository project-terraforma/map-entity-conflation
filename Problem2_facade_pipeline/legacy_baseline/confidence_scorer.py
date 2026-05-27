"""Assign interpretable confidence labels and scores for facade matches."""
from config import (
    ENTRANCE_PREFERRED_RADIUS_M,
    INSIDE_BUILDING_CONFIDENCE_PENALTY,
    MULTIPLE_CLOSE_FACADE_DELTA_M,
    NEAREST_FACADE_HIGH_THRESHOLD_M,
    NEAREST_FACADE_LOW_THRESHOLD_M,
    NEAREST_FACADE_MEDIUM_THRESHOLD_M,
    STREET_SUPPORT_RADIUS_M,
)


def distance_score(distance_m):
    """Convert distance to a bounded score without hiding the raw distance."""
    if distance_m is None:
        return 0.0
    if distance_m <= NEAREST_FACADE_HIGH_THRESHOLD_M:
        return 0.9
    if distance_m <= NEAREST_FACADE_MEDIUM_THRESHOLD_M:
        return 0.7
    if distance_m <= NEAREST_FACADE_LOW_THRESHOLD_M:
        return 0.35
    return 0.15


def score_match(
    distance_m,
    poi_inside_building=False,
    entrance_distance_m=None,
    street_distance_m=None,
    second_best_distance_m=None,
    has_building=True,
    invalid_geometry=False,
):
    """Return (score, label, evidence list) from the evidence available for one match."""
    evidence = []
    if not has_building:
        return 0.0, "needs_review_no_building_match", ["no_usable_building_match"]
    if invalid_geometry:
        return 0.0, "needs_review_invalid_geometry", ["invalid_building_or_facade_geometry"]

    score = distance_score(distance_m)
    if distance_m is not None:
        evidence.append(f"nearest_facade_distance_m={round(distance_m, 3)}")

    if (
        second_best_distance_m is not None
        and distance_m is not None
        and second_best_distance_m - distance_m <= MULTIPLE_CLOSE_FACADE_DELTA_M
    ):
        return min(score, 0.45), "needs_review_multiple_close_facades", evidence + ["multiple_close_facades"]

    if entrance_distance_m is not None and entrance_distance_m <= ENTRANCE_PREFERRED_RADIUS_M:
        return max(score, 0.93), "high_confidence_entrance_supported", evidence + [f"entrance_distance_m={round(entrance_distance_m, 3)}"]

    if street_distance_m is not None and street_distance_m <= STREET_SUPPORT_RADIUS_M:
        label = "medium_confidence_street_supported"
        if distance_m is not None and distance_m <= NEAREST_FACADE_HIGH_THRESHOLD_M and not poi_inside_building:
            label = "high_confidence_nearest_facade"
        return max(score, 0.72), label, evidence + [f"street_distance_m={round(street_distance_m, 3)}"]

    if poi_inside_building:
        adjusted = round(score * INSIDE_BUILDING_CONFIDENCE_PENALTY, 3)
        return adjusted, "medium_confidence_inside_building_nearest_edge", evidence + ["poi_inside_building"]

    if distance_m is not None and distance_m <= NEAREST_FACADE_HIGH_THRESHOLD_M:
        return score, "high_confidence_nearest_facade", evidence
    if distance_m is not None and distance_m <= NEAREST_FACADE_MEDIUM_THRESHOLD_M:
        return score, "medium_confidence_nearest_facade", evidence
    return score, "low_confidence_far_from_building", evidence
