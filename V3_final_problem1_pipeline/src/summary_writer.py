"""Write summary and manual evaluation outputs."""

import pandas as pd


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build summary_type,label,count rows for key V3 fields."""
    rows = [{"summary_type": "total_pois", "label": "total_pois", "count": int(len(df))}]
    summary_columns = [
        "final_label",
        "match_method",
        "street_connection_status",
        "poi_class",
        "building_status",
        "address_match_status",
    ]
    for column in summary_columns:
        if column not in df.columns:
            continue
        counts = df[column].fillna("missing").astype(str).value_counts()
        for label, count in counts.items():
            rows.append({"summary_type": column, "label": label, "count": int(count)})
    return pd.DataFrame(rows)


def save_summary(df: pd.DataFrame, output_path):
    """Save the pipeline summary CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_summary(df).to_csv(output_path, index=False)


def save_manual_evaluation_template(df: pd.DataFrame, output_path, sample_size, random_state=42):
    """Save a reproducible manual evaluation sample."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(sample_size, len(df))
    sample = df.sample(n=n, random_state=random_state).copy() if n else df.copy()
    sample["manual_result"] = ""
    sample["manual_notes"] = ""
    sample.to_csv(output_path, index=False)
