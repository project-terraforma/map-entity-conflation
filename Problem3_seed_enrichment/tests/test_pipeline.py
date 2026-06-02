"""Focused tests for imagery seed aggregation and evidence scoring."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from licensed_evidence_matcher import match_one as match_licensed_evidence
from local_overture_matcher import match_one
from raw_image_extractor import choose_business_name, filename_hint
from seed_builder import aggregate_seeds


class SeedBuilderTest(unittest.TestCase):
    def test_front_door_coordinates_are_preferred(self):
        observations = pd.DataFrame(
            [
                {"source_dataset": "boulder", "POI_name": "Demo Cafe", "overture_id": "", "Type": "sign", "Latitude": 40.0, "Longitude": -105.0, "image_filename": "sign.jpg", "Remarks": "sign", "source_row_number": 2, "source_file": "source.csv", "normalized_name": "demo cafe"},
                {"source_dataset": "boulder", "POI_name": "Demo Cafe", "overture_id": "", "Type": "front_door", "Latitude": 40.1, "Longitude": -105.1, "image_filename": "door.jpg", "Remarks": "door", "source_row_number": 3, "source_file": "source.csv", "normalized_name": "demo cafe"},
            ]
        )
        seeds = aggregate_seeds(observations)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds.iloc[0]["location_basis"], "front_door_median")
        self.assertEqual(seeds.iloc[0]["seed_lat"], 40.1)


class LocalMatcherTest(unittest.TestCase):
    def test_exact_nearby_name_is_confirmed(self):
        seed = {"poi_name_seed": "Demo Cafe", "seed_lat": 40.0, "seed_lon": -105.0}
        places = pd.DataFrame(
            [
                {
                    "local_overture_id": "abc",
                    "local_overture_name": "Demo Cafe",
                    "local_overture_lat": 40.00005,
                    "local_overture_lon": -105.00005,
                    "local_overture_address": "1 Main St",
                    "local_overture_category": "cafe",
                    "local_overture_websites": "",
                    "local_overture_phones": "",
                    "local_overture_operating_status": "open",
                }
            ]
        )
        self.assertEqual(match_one(seed, places)["local_overture_match_status"], "confirmed")


class RawImageExtractorTest(unittest.TestCase):
    def test_filename_hint_removes_capture_suffixes(self):
        self.assertEqual(filename_hint("chinook_pharmacy_PXL_20260128_195803460.jpg"), "Chinook Pharmacy")
        self.assertEqual(filename_hint("boulder-signs-03-05-2026_20260326_120603581.jpg"), "")

    def test_ocr_name_selection_prefers_supported_hint(self):
        lines = [
            {"text": "CHINOOK PHARMACY", "confidence": 0.92},
            {"text": "625 MAIN ST", "confidence": 0.88},
        ]
        name, confidence, basis = choose_business_name(lines, hint="Chinook Pharmacy")
        self.assertEqual(name, "CHINOOK PHARMACY")
        self.assertGreaterEqual(confidence, 0.72)
        self.assertEqual(basis, "vision_ocr")


class LicensedEvidenceScoringTest(unittest.TestCase):
    def test_verified_candidate_extracts_attributes_and_provenance(self):
        seed = {"poi_name_seed": "Demo Cafe", "seed_lat": 40.0, "seed_lon": -105.0}
        evidence = pd.DataFrame(
            [
                {
                    "provider_id": "licensed-1",
                    "name": "Demo Cafe",
                    "address": "1 Main St",
                    "lat": 40.00005,
                    "lon": -105.00005,
                    "category": "cafe",
                    "website": "https://example.test",
                    "source_url": "https://example.test/about",
                    "license": "permission-granted",
                }
            ]
        )
        result = match_licensed_evidence(seed, evidence)
        self.assertEqual(result["external_match_status"], "verified")
        self.assertEqual(result["external_website"], "https://example.test")
        self.assertEqual(result["external_license"], "permission-granted")


if __name__ == "__main__":
    unittest.main()
