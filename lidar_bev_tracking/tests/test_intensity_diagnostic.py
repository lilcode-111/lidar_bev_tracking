import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

from bev_tracking.error_codes import FrameStatus
from bev_tracking.intensity_diagnostic import (
    IntensityDiagnosticError,
    load_intensity_config,
    validate_i0_reproduces_baseline,
    validate_intensity_config_pair,
    validate_report_invariants,
)
from bev_tracking.result_types import FrameMetrics, FrameResult


def metric(tp=1, fp=2, fn=1, neutralized=1):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "per_class": {
            "car": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            }
        },
        "neutralized_detections": neutralized,
        "effective_car_detection_count": tp + fp + neutralized,
    }


def diagnostic_report(intensity_min=0.0, identical=True):
    metrics = {"0.50": metric(), "0.25": metric(tp=2, fp=1, fn=0, neutralized=1)}
    return {
        "source": {
            "requested_frame_ids": ["000001"],
            "diagnostic_manifest": {"sha256": "manifest-hash"},
        },
        "summary": {"geometry_failed_frames": 0},
        "frames": [
            {
                "frame_id": "000001",
                "failure_evidence": {
                    "parameters": {"intensity_min": intensity_min},
                    "summary": {
                        "stage_point_counts": {
                            "raw": 100,
                            "roi": 90,
                            "z_filter": 80,
                            "intensity_filter": 80 if identical else 70,
                        },
                        "z_to_intensity_identical": identical,
                        "num_positive_gt": 2,
                        "num_raw_detections": 5,
                        "num_car_detections_before_nms": 4,
                        "num_detections_after_nms": 4,
                        "metrics_by_iou": metrics,
                    },
                },
            }
        ],
    }


def manifest():
    return {"sha256": "manifest-hash", "num_frames": 1, "frame_ids": ["000001"]}


class IntensityConfigGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.i0 = load_intensity_config(root / "configs" / "experiments" / "v15" / "i0_intensity_038.yaml")
        cls.i1 = load_intensity_config(root / "configs" / "experiments" / "v15" / "i1_intensity_000.yaml")

    def test_i0_i1_configs_differ_only_by_intensity_min(self):
        differences = validate_intensity_config_pair(self.i0, self.i1)

        self.assertEqual(differences, [{"path": "detector.intensity_min", "i0": 0.38, "i1": 0.0}])

    def test_additional_algorithm_change_is_rejected(self):
        changed = copy.deepcopy(self.i1)
        changed["detector"]["eps"] = 0.7

        with self.assertRaises(IntensityDiagnosticError):
            validate_intensity_config_pair(self.i0, changed)


class IntensityInvariantTest(unittest.TestCase):
    def test_i1_requires_exact_z_and_intensity_point_identity(self):
        config = {"detector": {"intensity_min": 0.0}}
        result = validate_report_invariants(
            diagnostic_report(),
            config,
            manifest(),
            require_intensity_identity=True,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["z_to_intensity_identical_frames"], 1)
        self.assertEqual(result["metrics_by_iou"]["0.50"]["effective_car_detection_count"], 4)

    def test_i1_rejects_any_point_removed_by_intensity_stage(self):
        config = {"detector": {"intensity_min": 0.0}}

        with self.assertRaises(IntensityDiagnosticError):
            validate_report_invariants(
                diagnostic_report(identical=False),
                config,
                manifest(),
                require_intensity_identity=True,
            )

    def test_tp_plus_fn_conservation_is_required(self):
        report = diagnostic_report()
        report["frames"][0]["failure_evidence"]["summary"]["metrics_by_iou"]["0.50"]["fn"] = 5

        with self.assertRaises(IntensityDiagnosticError):
            validate_report_invariants(report, {"detector": {"intensity_min": 0.0}}, manifest())

    def test_i0_must_match_a0_prime_per_frame(self):
        report = diagnostic_report(intensity_min=0.38, identical=False)
        summary = report["frames"][0]["failure_evidence"]["summary"]
        baseline_metrics = {}
        for iou_key, values in summary["metrics_by_iou"].items():
            baseline_metrics[iou_key] = FrameMetrics(
                tp=values["tp"],
                fp=values["fp"],
                fn=values["fn"],
                precision=values["precision"],
                recall=values["recall"],
                f1=values["f1"],
                neutralized_detections=values["neutralized_detections"],
                per_class=values["per_class"],
            )
        baseline_frame = FrameResult(
            frame_id="000001",
            status=FrameStatus.SUCCESS,
            metrics_by_iou=baseline_metrics,
            num_positive_gt=2,
            num_raw_detections=5,
            num_car_detections_before_nms=4,
            num_detections_after_nms=4,
        )
        baseline_batch = SimpleNamespace(frame_results=[baseline_frame])

        result = validate_i0_reproduces_baseline(report, baseline_batch, manifest())

        self.assertTrue(result["passed"])
        self.assertEqual(result["num_frames_compared"], 1)


if __name__ == "__main__":
    unittest.main()
