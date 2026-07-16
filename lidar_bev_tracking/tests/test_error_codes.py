import json
import unittest

from bev_tracking.error_codes import BatchStatus, ErrorCode, ErrorStage, FrameStatus, is_metric_valid_status


class ErrorCodesTest(unittest.TestCase):
    def test_status_values_are_stable_strings(self):
        self.assertEqual(FrameStatus.SUCCESS.value, "success")
        self.assertEqual(FrameStatus.PARTIAL_SUCCESS.value, "partial_success")
        self.assertEqual(FrameStatus.SKIPPED.value, "skipped")
        self.assertEqual(FrameStatus.FAILED.value, "failed")
        self.assertEqual(BatchStatus.SUCCESS.value, "success")
        self.assertEqual(BatchStatus.PARTIAL_SUCCESS.value, "partial_success")
        self.assertEqual(BatchStatus.FAILED.value, "failed")

    def test_common_input_errors_are_distinct(self):
        self.assertEqual(ErrorCode.MISSING_BIN.value, "missing_bin")
        self.assertEqual(ErrorCode.MISSING_LABEL.value, "missing_label")
        self.assertEqual(ErrorCode.MISSING_CALIB.value, "missing_calib")
        self.assertNotEqual(ErrorCode.MISSING_BIN, ErrorCode.MISSING_LABEL)
        self.assertNotEqual(ErrorCode.MISSING_LABEL, ErrorCode.MISSING_CALIB)

    def test_error_stage_values_are_stable(self):
        self.assertEqual(ErrorStage.INPUT_CHECK.value, "input_check")
        self.assertEqual(ErrorStage.LABEL_PARSE.value, "label_parse")
        self.assertEqual(ErrorStage.EVALUATION.value, "evaluation")
        self.assertEqual(ErrorStage.REPORT_WRITE.value, "report_write")

    def test_metric_valid_status_mapping(self):
        self.assertTrue(is_metric_valid_status(FrameStatus.SUCCESS))
        self.assertTrue(is_metric_valid_status(FrameStatus.PARTIAL_SUCCESS))
        self.assertFalse(is_metric_valid_status(FrameStatus.SKIPPED))
        self.assertFalse(is_metric_valid_status(FrameStatus.FAILED))

    def test_enums_can_be_json_serialized_as_values(self):
        text = json.dumps({"status": FrameStatus.SUCCESS.value, "code": ErrorCode.MISSING_BIN.value})
        self.assertIn("success", text)
        self.assertIn("missing_bin", text)


if __name__ == "__main__":
    unittest.main()
