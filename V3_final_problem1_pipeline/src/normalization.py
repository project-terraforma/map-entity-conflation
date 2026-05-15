"""Text and address normalization helpers."""

import json
import re

import pandas as pd


STREET_ABBREVIATIONS = {
    "street": "st",
    "st": "st",
    "avenue": "ave",
    "ave": "ave",
    "road": "rd",
    "rd": "rd",
    "boulevard": "blvd",
    "blvd": "blvd",
    "drive": "dr",
    "dr": "dr",
    "lane": "ln",
    "ln": "ln",
    "court": "ct",
    "ct": "ct",
    "place": "pl",
    "pl": "pl",
    "parkway": "pkwy",
    "pkwy": "pkwy",
    "highway": "hwy",
    "hwy": "hwy",
}


def is_missing(value) -> bool:
    """Return True when a scalar value is null, empty, or string-null-like."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null"}
    return False


def stringify_value(value) -> str:
    """Convert nested Overture-ish values into readable text."""
    if is_missing(value):
        return ""
    if isinstance(value, dict):
        for key in ("common", "primary", "main", "value", "name", "freeform"):
            if key in value and not is_missing(value[key]):
                return stringify_value(value[key])
        return " ".join(stringify_value(v) for v in value.values() if not is_missing(v))
    if isinstance(value, (list, tuple, set)):
        return " ".join(stringify_value(v) for v in value if not is_missing(v))
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return stringify_value(json.loads(text))
            except json.JSONDecodeError:
                return text
        return text
    return str(value).strip()


def normalize_text(value) -> str:
    """Normalize casing, punctuation, spacing, and common street suffixes."""
    text = stringify_value(value).lower()
    if not text:
        return ""
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = []
    for token in text.split():
        tokens.append(STREET_ABBREVIATIONS.get(token, token))
    return " ".join(tokens)


def normalize_street(value) -> str:
    """Normalize a street name without a leading house number."""
    street = extract_street_from_address(value)
    return normalize_text(street)


def extract_house_number(address):
    """Extract the leading house number from a freeform address."""
    text = normalize_text(address)
    match = re.match(r"^(\d+[a-z]?)\b", text)
    return match.group(1) if match else ""


def extract_street_from_address(address):
    """Extract the street portion from a freeform address."""
    if is_missing(address):
        return None
    text = stringify_value(address).strip()
    text = re.sub(r"^\s*\d+[A-Za-z]?\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def build_address_text(row):
    """Build a display address from address fields."""
    freeform = row.get("freeform") or row.get("address") or row.get("address_text")
    if not is_missing(freeform):
        return stringify_value(freeform).strip()
    number = stringify_value(row.get("number") or row.get("address_number")).strip()
    street = stringify_value(row.get("street") or row.get("address_street")).strip()
    return " ".join(part for part in [number, street] if part).strip()
