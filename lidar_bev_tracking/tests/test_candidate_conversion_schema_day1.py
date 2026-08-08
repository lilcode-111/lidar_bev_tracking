import unittest

from bev_tracking.candidate_conversion import (
    CANDIDATE_CONVERSION_SCHEMA_VERSION,
    CANDIDATE_CONVERSION_SOURCE_OF_TRUTH,
    build_candidate_conversion_report,
    validate_candidate_conversion_payload,
)


class CandidateConversionSchemaDay1Test(unittest.TestCase):
    def test_schema_metadata_declares_single_source_of_truth(self):
        payload = build_candidate_conversion_report(
            frame_id="1",
            gt_boxes=[],
            stages={},
            clusters=[],
            raw_detections=[],
            detections_after_nms=[],
            evaluation={"matches": []},
            variant="C0",
            min_points=20,
        )
        self.assertEqual(payload["schema_version"], CANDIDATE_CONVERSION_SCHEMA_VERSION)
        self.assertEqual(payload["source_of_truth"], CANDIDATE_CONVERSION_SOURCE_OF_TRUTH)
        self.assertIn("gt_candidate_records", payload["deprecated_fields"])
        self.assertEqual(validate_candidate_conversion_payload(payload), True)

    def test_duplicate_gt_evidence_is_rejected(self):
        payload = {
            "schema_version": CANDIDATE_CONVERSION_SCHEMA_VERSION,
            "source_of_truth": CANDIDATE_CONVERSION_SOURCE_OF_TRUTH,
            "num_positive_gt": 2,
            "evidence": [
                {"frame_id": "000001", "gt_id": "gt_1"},
                {"frame_id": "000001", "gt_id": "gt_1"},
            ],
        }
        with self.assertRaises(ValueError):
            validate_candidate_conversion_payload(payload)

    def test_evidence_count_must_match_positive_gt_count(self):
        payload = {
            "schema_version": CANDIDATE_CONVERSION_SCHEMA_VERSION,
            "source_of_truth": CANDIDATE_CONVERSION_SOURCE_OF_TRUTH,
            "num_positive_gt": 2,
            "evidence": [{"frame_id": "000001", "gt_id": "gt_1"}],
        }
        with self.assertRaises(ValueError):
            validate_candidate_conversion_payload(payload)


if __name__ == "__main__":
    unittest.main()
