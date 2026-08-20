import tempfile
import unittest
from pathlib import Path

from bev_tracking.error_codes import FrameStatus
from bev_tracking.pipeline import run_kitti_bev_evaluation, run_kitti_frame_evaluation
from bev_tracking.result_types import FrameResult
from bev_tracking.synthetic import generate_frame


def write_minimal_calib(calib_path):
    calib_path.write_text(
        "\n".join(
            [
                "P0: 0 0 0 0 0 0 0 0 0 0 0 0",
                "P1: 0 0 0 0 0 0 0 0 0 0 0 0",
                "P2: 0 0 0 0 0 0 0 0 0 0 0 0",
                "P3: 0 0 0 0 0 0 0 0 0 0 0 0",
                "R0_rect: 1 0 0 0 1 0 0 0 1",
                "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_kitti_label(label_path):
    label_path.write_text(
        "\n".join(
            [
                "Car 0.00 0 0.00 0 0 50 50 1.50 1.90 4.50 3.00 0.75 12.00 0.15",
                "Car 0.00 0 0.00 0 0 50 50 1.50 2.00 4.70 -5.00 0.75 24.00 -0.25",
                "Pedestrian 0.00 0 0.00 0 0 50 50 1.70 0.80 0.80 7.50 0.85 18.00 0.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def make_mini_kitti(root, frame_id="000000"):
    data_root = Path(root) / "kitti"
    velodyne_dir = data_root / "training" / "velodyne"
    label_dir = data_root / "training" / "label_2"
    calib_dir = data_root / "training" / "calib"
    velodyne_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    calib_dir.mkdir(parents=True)

    points, _ = generate_frame(seed=7)
    points.astype("float32").tofile(velodyne_dir / f"{frame_id}.bin")
    write_kitti_label(label_dir / f"{frame_id}.txt")
    write_minimal_calib(calib_dir / f"{frame_id}.txt")
    return data_root


class FramePipelineResultTest(unittest.TestCase):
    def test_frame_pipeline_returns_success_frame_result_without_writing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = make_mini_kitti(tmp)

            result = run_kitti_frame_evaluation(data_root=data_root, frame_id="000000", oriented=True)

            self.assertIsInstance(result, FrameResult)
            self.assertEqual(result.status, FrameStatus.SUCCESS)
            self.assertTrue(result.metric_valid)
            self.assertIn("0.50", result.metrics_by_iou)
            self.assertIn("0.25", result.metrics_by_iou)
            self.assertIn("car", result.metrics_by_iou["0.50"].per_class)
            self.assertGreater(result.num_points, 0)
            self.assertEqual(result.num_labels_raw, 3)
            self.assertIsNotNone(result.load_time_ms)
            self.assertIsNotNone(result.total_time_ms)
            self.assertFalse((Path(tmp) / "outputs").exists())

            gesr_result = run_kitti_frame_evaluation(
                data_root=data_root,
                frame_id="000000",
                oriented=True,
                gesr_enabled=True,
            )
            self.assertEqual(gesr_result.status, FrameStatus.SUCCESS)
            self.assertTrue(gesr_result.artifacts["gesr"]["enabled"])
            self.assertIsNotNone(gesr_result.artifacts["gesr"]["runtime_evidence"])
            self.assertIn("obstacle_source_indices", gesr_result.artifacts["gesr"])

    def test_pipeline_missing_inputs_are_skipped_not_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = make_mini_kitti(tmp)
            missing_bin = data_root / "training" / "velodyne" / "000000.bin"
            missing_bin.unlink()

            result = run_kitti_frame_evaluation(data_root=data_root, frame_id="000000")

            self.assertEqual(result.status, FrameStatus.SKIPPED)
            self.assertFalse(result.metric_valid)

    def test_legacy_pipeline_still_writes_json_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = make_mini_kitti(tmp)
            report_dir = Path(tmp) / "reports"

            report, output_path = run_kitti_bev_evaluation(
                data_root=data_root,
                frame_id="000000",
                oriented=False,
                report_dir=report_dir,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(report["status"], "success")
            self.assertTrue(report["metric_valid"])
            self.assertIn("metrics_by_iou", report)


if __name__ == "__main__":
    unittest.main()
