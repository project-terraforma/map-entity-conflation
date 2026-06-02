"""Extract reviewable POI seeds from unannotated street-level imagery."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from config import PROBLEM3_ROOT, RAW_EXTRACTIONS_OUTPUT, RAW_REVIEW_QUEUE_OUTPUT
from text_utils import clean_text, normalize_text, similarity

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff"}
GENERIC_FILENAME_PREFIXES = {"boulder signs", "image", "img", "pxl", "photo", "dsc", "sign census"}
FILENAME_STOP_TOKENS = {
    "awning",
    "back",
    "corner",
    "census",
    "door",
    "east",
    "entrance",
    "exterior",
    "front",
    "left",
    "main",
    "north",
    "parking",
    "rear",
    "right",
    "sign",
    "south",
    "up",
    "west",
    "window",
}
MANIFEST_COLUMNS = {"image_filename", "latitude", "longitude", "source_dataset", "poi_name_override", "address_hint"}
VISION_SOURCE = PROBLEM3_ROOT / "tools" / "vision_ocr.m"
VISION_BINARY = PROBLEM3_ROOT / "outputs" / "vision_ocr"
DEFAULT_ACCEPT_CONFIDENCE = 0.72


def list_images(images_dir):
    """Return supported image paths in stable filename order."""
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"Images directory does not exist: {root}")
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def load_capture_manifest(path=None):
    """Load optional capture metadata and reviewer overrides keyed by filename."""
    if not path:
        return {}
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Capture manifest does not exist: {manifest_path}")
    df = pd.read_csv(manifest_path)
    if "image_filename" not in df.columns:
        raise ValueError("Capture manifest must contain image_filename.")
    for column in MANIFEST_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return {clean_text(row["image_filename"]): row.to_dict() for _, row in df.iterrows()}


def filename_hint(path):
    """Infer a weak business-name hint from descriptive image filenames."""
    stem = re.sub(r"[_-]+", " ", Path(path).stem.lower())
    stem = re.sub(r"\b(?:19|20)\d{6,}\b", " ", stem)
    stem = re.sub(r"\b\d{1,2} \d{1,2} \d{4}\b", " ", stem)
    stem = re.sub(r"\bpxl\b", " ", stem)
    tokens = stem.split()
    if not tokens:
        return ""
    if "census" in tokens:
        tokens = tokens[: tokens.index("census")]
    while tokens and (tokens[-1] in FILENAME_STOP_TOKENS or tokens[-1].isdigit()):
        tokens.pop()
    while tokens and tokens[0] in FILENAME_STOP_TOKENS:
        tokens.pop(0)
    hint = " ".join(tokens).strip()
    if not hint or any(hint.startswith(prefix) for prefix in GENERIC_FILENAME_PREFIXES):
        return ""
    return " ".join(token.capitalize() for token in hint.split())


def _valid_ocr_line(text):
    text = clean_text(text)
    if len(text) < 2 or len(text) > 80:
        return False
    if sum(char.isalpha() for char in text) < 2:
        return False
    lowered = text.lower()
    if "http" in lowered or "www." in lowered or "@" in lowered:
        return False
    return True


def _line_score(text, confidence, hint=""):
    """Score OCR text as a plausible storefront name."""
    normalized = normalize_text(text)
    tokens = normalized.split()
    if not tokens:
        return 0.0
    score = 0.72 * float(confidence)
    if 1 <= len(tokens) <= 5:
        score += 0.12
    if hint:
        score += 0.16 * similarity(text, hint)
    if any(char.isdigit() for char in text):
        score -= 0.18
    if len(normalized) > 55:
        score -= 0.10
    return max(0.0, min(1.0, score))


def choose_business_name(lines, hint="", override=""):
    """Choose a machine seed name while retaining all OCR text for review."""
    if clean_text(override):
        return clean_text(override), 1.0, "manifest_override"
    candidates = []
    for line in lines:
        text = clean_text(line.get("text"))
        if not _valid_ocr_line(text):
            continue
        score = _line_score(text, line.get("confidence", 0.0), hint)
        candidates.append((score, text))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if candidates and candidates[0][0] >= 0.55:
        return candidates[0][1], round(candidates[0][0], 6), "vision_ocr"
    if hint:
        return hint, 0.55, "filename_hint"
    return "", 0.0, "unresolved"


def run_vision_ocr(images):
    """Run Apple Vision OCR for a batch of images using a bundled Objective-C helper."""
    if not shutil.which("xcrun"):
        raise RuntimeError("Apple Vision OCR requires macOS Command Line Tools (xcrun).")
    if not VISION_SOURCE.exists():
        raise FileNotFoundError(f"Vision OCR helper is missing: {VISION_SOURCE}")
    if not VISION_BINARY.exists() or VISION_BINARY.stat().st_mtime < VISION_SOURCE.stat().st_mtime:
        VISION_BINARY.parent.mkdir(parents=True, exist_ok=True)
        compile_command = [
            "xcrun",
            "clang",
            "-fobjc-arc",
            "-fblocks",
            "-framework",
            "Foundation",
            "-framework",
            "ImageIO",
            "-framework",
            "CoreGraphics",
            "-framework",
            "Vision",
            str(VISION_SOURCE),
            "-o",
            str(VISION_BINARY),
        ]
        compiled = subprocess.run(compile_command, text=True, capture_output=True, encoding="utf-8", errors="replace")
        if compiled.returncode != 0:
            raise RuntimeError(f"Could not compile Apple Vision OCR helper:\n{compiled.stderr.strip() or compiled.stdout.strip()}")
    command = [str(VISION_BINARY), *[str(path.resolve()) for path in images]]
    result = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"Apple Vision OCR failed:\n{result.stderr.strip() or result.stdout.strip()}")
    rows = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows[str(Path(payload["image_path"]).resolve())] = payload
    return rows


def _float_or_blank(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(numeric) else float(numeric)


def extract_images(images_dir, manifest_path=None, max_images=None, accept_confidence=DEFAULT_ACCEPT_CONFIDENCE):
    """Extract OCR, locations, and machine seed names from raw imagery."""
    images = list_images(images_dir)
    if max_images is not None:
        images = images[: int(max_images)]
    manifest = load_capture_manifest(manifest_path)
    vision_results = run_vision_ocr(images)
    rows = []
    for image_path in images:
        payload = vision_results.get(str(image_path.resolve()), {})
        metadata = manifest.get(image_path.name, {})
        lat = _float_or_blank(metadata.get("latitude"))
        lon = _float_or_blank(metadata.get("longitude"))
        location_basis = "capture_manifest"
        if lat == "" or lon == "":
            lat = _float_or_blank(payload.get("latitude"))
            lon = _float_or_blank(payload.get("longitude"))
            location_basis = "exif_gps" if lat != "" and lon != "" else "missing"
        hint = filename_hint(image_path)
        lines = payload.get("lines") or []
        name, confidence, name_basis = choose_business_name(lines, hint=hint, override=metadata.get("poi_name_override"))
        accepted = bool(name and lat != "" and lon != "" and confidence >= float(accept_confidence))
        rows.append(
            {
                "image_filename": image_path.name,
                "image_path": str(image_path.resolve()),
                "source_dataset": clean_text(metadata.get("source_dataset")) or "raw_imagery",
                "poi_name_seed": name,
                "extraction_confidence": confidence,
                "name_basis": name_basis,
                "filename_hint": hint,
                "latitude": lat,
                "longitude": lon,
                "location_basis": location_basis,
                "address_hint": clean_text(metadata.get("address_hint")),
                "ocr_text": " | ".join(clean_text(line.get("text")) for line in lines if clean_text(line.get("text"))),
                "ocr_lines_json": json.dumps(lines, ensure_ascii=True),
                "extraction_error": clean_text(payload.get("error")),
                "accepted_as_seed": accepted,
                "review_reason": "" if accepted else _review_reason(name, confidence, lat, lon, accept_confidence),
            }
        )
    return pd.DataFrame(rows)


def _review_reason(name, confidence, lat, lon, accept_confidence):
    reasons = []
    if not name:
        reasons.append("no_business_name")
    elif confidence < float(accept_confidence):
        reasons.append("low_name_confidence")
    if lat == "" or lon == "":
        reasons.append("missing_location")
    return " | ".join(reasons)


def write_extraction_outputs(extractions, output_path=RAW_EXTRACTIONS_OUTPUT, review_path=RAW_REVIEW_QUEUE_OUTPUT):
    """Write the full OCR audit table and rows needing review."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    extractions.to_csv(output_path, index=False)
    extractions[~extractions["accepted_as_seed"]].to_csv(review_path, index=False)
