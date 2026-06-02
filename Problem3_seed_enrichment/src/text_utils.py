"""Small normalization and serialization helpers."""

import ast
import json
import math
import re
from difflib import SequenceMatcher


def clean_text(value):
    """Return a safe scalar string."""
    if value is None:
        return ""
    try:
        if math.isnan(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_text(value):
    """Normalize a POI name for grouping and fuzzy comparison."""
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def similarity(left, right):
    """Return SequenceMatcher similarity for normalized names."""
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def parse_nested(value):
    """Parse JSON-ish values emitted by GeoJSON and Parquet readers."""
    if isinstance(value, (dict, list, tuple)):
        return value
    text = clean_text(value)
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return value


def first_nested_text(value, preferred_keys=("primary", "common", "freeform", "text")):
    """Extract a representative text value from nested Overture-ish fields."""
    parsed = parse_nested(value)
    if isinstance(parsed, dict):
        for key in preferred_keys:
            if key in parsed:
                extracted = first_nested_text(parsed[key], preferred_keys)
                if extracted:
                    return extracted
        for nested_value in parsed.values():
            extracted = first_nested_text(nested_value, preferred_keys)
            if extracted:
                return extracted
        return ""
    if isinstance(parsed, (list, tuple)):
        for nested_value in parsed:
            extracted = first_nested_text(nested_value, preferred_keys)
            if extracted:
                return extracted
        return ""
    return clean_text(parsed)


def join_nested_text(value):
    """Serialize list-like Overture attributes for CSV output."""
    parsed = parse_nested(value)
    if isinstance(parsed, dict):
        return json.dumps(parsed, sort_keys=True)
    if isinstance(parsed, (list, tuple)):
        return " | ".join(clean_text(item) for item in parsed if clean_text(item))
    return clean_text(parsed)

