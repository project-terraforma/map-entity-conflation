"""Run the standalone V3 Problem 1 conflation pipeline."""

from pathlib import Path

from address_matcher import match_pois_to_addresses
from building_matcher import match_buildings
from building_validator import validate_building_relationships
from config import (
    FINAL_OUTPUT,
    FINAL_OUTPUT_COLUMNS,
    MANUAL_EVAL_OUTPUT,
    MANUAL_EVAL_SAMPLE_SIZE,
    OUTPUT_DIR,
    SUMMARY_OUTPUT,
)
from data_loader import load_raw_layers
from labeler import enrich_labels
from street_connector import add_street_connection
from summary_writer import save_manual_evaluation_template, save_summary


def build_final_dataframe(df):
    """Return final output columns in the requested order."""
    result = df.copy()
    for column in FINAL_OUTPUT_COLUMNS:
        if column not in result.columns:
            result[column] = None
    return result[FINAL_OUTPUT_COLUMNS].copy()


def print_terminal_summary(df, output_paths):
    """Print a readable terminal summary."""
    print("")
    print("Final Labels Summary:")
    print(df["final_label"].value_counts(dropna=False).to_string())
    print("")
    print(f"Total rows: {len(df)}")
    print("Output file paths:")
    for path in output_paths:
        print(f"- {Path(path).resolve()}")


def main():
    """Run all V3 stages end-to-end from raw Overture inputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        places, addresses, buildings, streets = load_raw_layers()
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(str(exc))
        raise SystemExit(1) from exc

    matched = match_pois_to_addresses(places, addresses)
    with_buildings = match_buildings(matched, buildings)
    validated = validate_building_relationships(with_buildings)
    labeled = enrich_labels(validated)
    with_streets = add_street_connection(labeled, streets)

    final_df = build_final_dataframe(with_streets)
    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(FINAL_OUTPUT, index=False)
    save_summary(final_df, SUMMARY_OUTPUT)
    save_manual_evaluation_template(
        final_df,
        MANUAL_EVAL_OUTPUT,
        sample_size=MANUAL_EVAL_SAMPLE_SIZE,
        random_state=42,
    )

    print(f"Final Labels: {final_df['final_label'].notna().sum()}")
    print_terminal_summary(final_df, [FINAL_OUTPUT, SUMMARY_OUTPUT, MANUAL_EVAL_OUTPUT])


if __name__ == "__main__":
    main()
