import json
import tempfile
import unittest
from pathlib import Path

from bev_tracking.gesr_phase2_gate import analyze_phase2_runs


VARIANTS = ("T0", "T2", "GESR-v1")


class GESRPhase2GateTest(unittest.TestCase):
    def test_gate0_and_gate_a_pass_from_minimal_frame_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_ids = [f"{index:06d}" for index in range(25)]
            identities = [(frame_ids[index % 11], f"gt_{index + 1}") for index in range(22)]
            identity_path = root / "identity.json"
            identity_path.write_text(
                json.dumps(
                    {
                        "diagnostic_25": {"ordered_frame_ids": frame_ids},
                        "delta_22": {"ordered_identity_list": identities},
                    }
                ),
                encoding="utf-8",
            )
            gate_path = root / "gate.json"
            gate_path.write_text(
                json.dumps(
                    {
                        "gates": {
                            "GateA": {
                                "material_recovery_count_min": 6,
                                "median_iou_gain_vs_T0_min": 0.05,
                                "iou_ge_0_25_count_gain_vs_T0_min": 4,
                                "material_regression_count_max": 2,
                                "material_recovery_frame_count_min": 3,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            run_dirs = {}
            for variant in VARIANTS:
                run_dir = root / variant
                frames_dir = run_dir / "frames"
                frames_dir.mkdir(parents=True)
                run_dirs[variant] = run_dir
                (run_dir / "config_effective.json").write_text(
                    json.dumps(
                        {
                            "phase2": {"variant": variant},
                            "detector": {
                                "intensity_min": 0.15 if variant == "T2" else 0.38,
                                "gesr_enabled": variant == "GESR-v1",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "run": {"batch_status": "success"},
                            "reproducibility": {
                                "git_commit": "a" * 40,
                                "git_dirty": False,
                                "requested_frame_ids": frame_ids,
                            },
                            "frames": {
                                "requested": 25,
                                "success": 25,
                                "metric_valid": 25,
                                "failed": 0,
                                "skipped": 0,
                            },
                            "metrics_by_iou": {"0.50": {}, "0.25": {}},
                        }
                    ),
                    encoding="utf-8",
                )

                records_by_frame = {frame_id: [] for frame_id in frame_ids}
                for index, (frame_id, gt_id) in enumerate(identities):
                    if variant == "T0":
                        iou = 0.20
                    elif variant == "T2":
                        iou = 0.30
                    elif index < 6:
                        iou = 0.32
                    elif index < 12:
                        iou = 0.26
                    else:
                        iou = 0.20
                    records_by_frame[frame_id].append(
                        {"frame_id": frame_id, "gt_id": gt_id, "variant": variant, "pca_iou": iou}
                    )
                for frame_id, records in records_by_frame.items():
                    artifacts = {
                        "phase2_delta22_geometry": {"variant": variant, "records": records}
                    }
                    if variant == "GESR-v1":
                        artifacts["gesr"] = {
                            "runtime_evidence": {"invariants": {"identity": True}}
                        }
                    (frames_dir / f"{frame_id}.json").write_text(
                        json.dumps({"artifacts": artifacts}), encoding="utf-8"
                    )

            result = analyze_phase2_runs(run_dirs, identity_path, gate_path)

            self.assertEqual(result["Gate0_Phase2"]["result"], "PASS")
            self.assertEqual(result["Gate_A"]["result"], "PASS")
            self.assertEqual(result["Gate_A"]["metrics"]["material_recovery_count"], 6)
            self.assertEqual(result["Gate_A"]["metrics"]["material_regression_count"], 0)
            self.assertEqual(len(result["Gate_A"]["delta22"]), 22)

    def test_gate_a_is_not_evaluated_when_gate0_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_ids = [f"{i:06d}" for i in range(25)]
            identity_path = root / "identity.json"
            identity_path.write_text(
                json.dumps(
                    {
                        "diagnostic_25": {"ordered_frame_ids": frame_ids},
                        "delta_22": {"ordered_identity_list": []},
                    }
                ),
                encoding="utf-8",
            )
            gate_path = root / "gate.json"
            gate_path.write_text(
                json.dumps(
                    {
                        "gates": {
                            "GateA": {
                                "material_recovery_count_min": 6,
                                "median_iou_gain_vs_T0_min": 0.05,
                                "iou_ge_0_25_count_gain_vs_T0_min": 4,
                                "material_regression_count_max": 2,
                                "material_recovery_frame_count_min": 3,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            run_dirs = {}
            for variant in VARIANTS:
                run_dir = root / variant
                (run_dir / "frames").mkdir(parents=True)
                run_dirs[variant] = run_dir
                (run_dir / "config_effective.json").write_text(
                    json.dumps(
                        {
                            "phase2": {"variant": variant},
                            "detector": {
                                "intensity_min": 0.15 if variant == "T2" else 0.38,
                                "gesr_enabled": variant == "GESR-v1",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "run": {"batch_status": "failed"},
                            "reproducibility": {
                                "git_commit": "a" * 40,
                                "git_dirty": False,
                                "requested_frame_ids": [],
                            },
                            "frames": {},
                            "metrics_by_iou": {},
                        }
                    ),
                    encoding="utf-8",
                )
            result = analyze_phase2_runs(run_dirs, identity_path, gate_path)
            self.assertEqual(result["Gate0_Phase2"]["result"], "FAIL")
            self.assertEqual(result["Gate_A"]["result"], "NOT_EVALUATED")


if __name__ == "__main__":
    unittest.main()
