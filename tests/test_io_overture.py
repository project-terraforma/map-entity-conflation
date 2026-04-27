from __future__ import annotations

import pandas as pd

from src.io_overture import _build_remote_bbox_sql, _flatten_struct_columns, _remote_path_candidates


def test_flatten_struct_columns_creates_dotted_columns() -> None:
    df = pd.DataFrame(
        {
            "id": ["x1"],
            "names": [{"primary": "Cafe Blue"}],
            "bbox": [{"xmin": -105.3, "ymin": 39.9, "xmax": -105.2, "ymax": 40.0}],
        }
    )

    flattened = _flatten_struct_columns(df)
    assert flattened.loc[0, "names.primary"] == "Cafe Blue"
    assert flattened.loc[0, "bbox.xmin"] == -105.3
    assert flattened.loc[0, "bbox.ymax"] == 40.0


def test_build_remote_bbox_sql_supports_struct_bbox() -> None:
    sql = _build_remote_bbox_sql(["id", "bbox", "geometry"], (-105.3, 39.9, -105.2, 40.0))
    assert sql is not None
    assert "bbox.xmax" in sql
    assert "bbox.ymin" in sql


def test_build_remote_bbox_sql_supports_flat_bbox_columns() -> None:
    sql = _build_remote_bbox_sql(
        ["id", "bbox.xmin", "bbox.ymin", "bbox.xmax", "bbox.ymax"],
        (-105.3, 39.9, -105.2, 40.0),
    )
    assert sql is not None
    assert '"bbox.xmin"' in sql


def test_remote_path_candidates_include_overture_fallbacks() -> None:
    candidates = _remote_path_candidates(
        "s3://overturemaps-us-west-2/release/2026-03-18.0/theme=places/type=place/*"
    )
    assert candidates[0].endswith("/type=place/*")
    assert any(candidate.endswith("/type=place/*.parquet") for candidate in candidates)
    assert any("/theme=places/*/*.parquet" in candidate for candidate in candidates)
