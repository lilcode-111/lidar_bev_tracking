import tempfile
import unittest
from pathlib import Path

from bev_tracking.batch_pipeline import run_kitti_batch_frame_results
from bev_tracking.error_codes import ErrorCode, FrameStatus
from bev_tracking.synthetic import generate_frame


def write_calib(calib_path):
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


def write_label(label_path):
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


def make_frame(data_root, frame_id, with_bin=True, with_label=True, with_calib=True, empty_label=False):
    frame_id = str(frame_id).zfill(6)
    velodyne_dir = data_root / "training" / "velodyne"
    label_dir = data_root / "training" / "label_2"
    calib_dir = data_root / "training" / "calib"
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)

    if with_bin:
        points, _ = generate_frame(seed=7 + int(frame_id))
        points.astype("float32").tofile(velodyne_dir / f"{frame_id}.bin")
    if with_label:
        label_path = label_dir / f"{frame_id}.txt"
        if empty_label:
            label_path.write_text("", encoding="utf-8")
        else:
            write_label(label_path)
    if with_calib:
        write_calib(calib_dir / f"{frame_id}.txt")


class BatchMissingInputsTest(unittest.TestCase):
    def test_missing_inputs_become_skipped_frame_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "kitti"
            make_frame(data_root, "000000", with_bin=False)
            make_frame(data_root, "000001", with_label=False)
            make_frame(data_root, "000002", with_calib=False)

            results = run_kitti_batch_frame_results(data_root=data_root, frame_ids=["000000", "000001", "000002"])

            self.assertEqual([result.status for result in results], [FrameStatus.SKIPPED] * 3)
            self.assertEqual([result.error.error_code for result in results], [
                ErrorCode.MISSING_BIN,
                ErrorCode.MISSING_LABEL,
                ErrorCode.MISSING_CALIB,
            ])
            self.assertTrue(all(not result.metric_valid for result in results))
            self.assertTrue(all(result.metrics_by_iou == {} for result in results))

    def test_empty_label_file_is_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "kitti"
            make_frame(data_root, "000000", empty_label=True)

            result = run_kitti_batch_frame_results(data_root=data_root, frame_ids=["000000"])[0]

            self.assertEqual(result.status, FrameStatus.SUCCESS)
            self.assertTrue(result.metric_valid)
            self.assertEqual(result.num_labels_raw, 0)
            self.assertIn("0.50", result.metrics_by_iou)

    def test_missing_middle_frame_does_not_stop_later_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "kitti"
            make_frame(data_root, "000000")
            make_frame(data_root, "000002")

            results = run_kitti_batch_frame_results(data_root=data_root, frame_ids=["000000", "000001", "000002"])

            self.assertEqual([result.frame_id for result in results], ["000000", "000001", "000002"])
            self.assertEqual(results[0].status, FrameStatus.SUCCESS)
            self.assertEqual(results[1].status, FrameStatus.SKIPPED)
            self.assertEqual(results[1].error.error_code, ErrorCode.MISSING_BIN)
            self.assertEqual(results[2].status, FrameStatus.SUCCESS)


if __name__ == "__main__":
    unittest.main()
