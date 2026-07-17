import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bev_tracking.batch_pipeline as batch_pipeline
from bev_tracking.error_codes import ErrorCode, FrameStatus
from tests.test_batch_missing_inputs import make_frame


class BatchErrorIsolationTest(unittest.TestCase):
    def test_unexpected_frame_error_does_not_stop_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "kitti"
            make_frame(data_root, "000000")
            make_frame(data_root, "000001")
            make_frame(data_root, "000002")
            original = batch_pipeline.run_kitti_frame_evaluation

            def fake_run_kitti_frame_evaluation(*args, **kwargs):
                if str(kwargs["frame_id"]).zfill(6) == "000001":
                    raise RuntimeError("simulated detector crash")
                return original(*args, **kwargs)

            with mock.patch.object(batch_pipeline, "run_kitti_frame_evaluation", side_effect=fake_run_kitti_frame_evaluation):
                results = batch_pipeline.run_kitti_batch_frame_results(
                    data_root=data_root,
                    frame_ids=["000000", "000001", "000002"],
                )

        self.assertEqual([result.frame_id for result in results], ["000000", "000001", "000002"])
        self.assertEqual(results[0].status, FrameStatus.SUCCESS)
        self.assertEqual(results[1].status, FrameStatus.FAILED)
        self.assertEqual(results[1].error.error_code, ErrorCode.UNEXPECTED_FRAME_ERROR)
        self.assertEqual(results[1].error.exception_type, "RuntimeError")
        self.assertEqual(results[2].status, FrameStatus.SUCCESS)
        self.assertFalse(results[1].metric_valid)
        self.assertEqual(results[1].metrics_by_iou, {})


if __name__ == "__main__":
    unittest.main()
