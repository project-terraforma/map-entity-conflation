"""Validate POI/address matches using building evidence."""

import pandas as pd

from config import CLUSTERED_ADDRESS_MIN_POIS


def validate_building_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """Add building validation labels from match status, clusters, and geometry."""
    result = df.copy()
    address_counts = result["matched_address_text"].value_counts(dropna=True)
    result["address_cluster_size"] = result["matched_address_text"].map(address_counts).fillna(0).astype(int)

    def label(row):
        status = row.get("address_match_status")
        building_status = row.get("building_status")
        relation = row.get("real_building_relation")
        cluster_size = row.get("address_cluster_size", 0)

        if relation == "same_building" and status in {"matched_high", "matched_medium"}:
            return "high_confidence_strong"
        if building_status == "building_consistent" and status in {"matched_high", "matched_medium"}:
            return "high_confidence_strong"
        if status in {"matched_high", "matched_medium"} and building_status in {"building_possible", "building_unknown"}:
            return "medium_confidence_valid"
        if cluster_size >= CLUSTERED_ADDRESS_MIN_POIS and status in {"matched_medium", "uncertain"}:
            return "clustered_commercial_area"
        if relation == "different_building" or building_status == "building_conflict":
            return "needs_review"
        if status == "uncertain":
            return "needs_review"
        return "low_confidence"

    result["building_validation_label"] = result.apply(label, axis=1)
    return result
