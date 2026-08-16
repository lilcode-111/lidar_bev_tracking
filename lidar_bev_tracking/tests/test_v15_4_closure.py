import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bev_tracking.v15_4_closure import (
    DEFAULT_ARTIFACT_PATHS,
    V154ClosureError,
    build_v15_4_closure,
    validate_v15_4_closure,
)


class V154ClosureTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = {
            artifact_id: f"artifacts/{artifact_id}.json"
            for artifact_id in DEFAULT_ARTIFACT_PATHS
        }
        self._write_fixture_artifacts()

    def tearDown(self):
        self.temp.cleanup()

    def test_builds_resolvable_evidence_and_recomputes_frozen_selection(self):
        _, closure, validation = build_v15_4_closure(
            repo_root=self.root,
            artifact_paths=self.paths,
        )
        self.assertEqual(validation["status"], "PASS")
        self.assertTrue(validation["all_registry_hashes_verified"])
        self.assertTrue(validation["all_gate0_refs_resolved"])
        self.assertTrue(validation["all_gate0_fields_resolved"])
        self.assertTrue(validation["formal_commit_identity_verified"])
        self.assertTrue(validation["source_point_monotonicity_verified"])
        self.assertTrue(validation["result_selection_recomputed"])
        self.assertFalse(validation["formal_results_modified"])
        self.assertFalse(validation["formal_100_rerun"])
        conclusion = closure["frozen_conclusion"]
        self.assertTrue(conclusion["experiment_valid"])
        self.assertTrue(conclusion["root_cause_intervention_validated"])
        self.assertFalse(conclusion["release_goal_achieved"])
        self.assertEqual(conclusion["qualified_release_candidates"], [])
        self.assertIsNone(conclusion["selected_release_candidate"])
        self.assertEqual(conclusion["safe_configuration"], "T0")
        self.assertEqual(conclusion["safe_intensity_min"], 0.38)
        self.assertEqual(closure["product_review"]["status"], "PENDING")

    def test_modified_registered_artifact_is_rejected(self):
        build_v15_4_closure(repo_root=self.root, artifact_paths=self.paths)
        self._write("formal_experiment", {"schema_version": "tampered"})
        with self.assertRaisesRegex(V154ClosureError, "registry_hashes"):
            validate_v15_4_closure(repo_root=self.root)

    def _write_fixture_artifacts(self):
        commit = "a" * 40
        diagnostic_hash = "b" * 64
        formal_hash = "c" * 64
        delta_hash = "d" * 64
        self._write("pre_run_identity", {
            "schema_version": "15.4-pre-run-identity-v1",
            "diagnostic_25": {"manifest_sha256": diagnostic_hash},
            "formal_100": {"manifest_sha256": formal_hash},
            "delta_22": {"ordered_identity_sha256": delta_hash},
            "t0_100_reference": {"intensity_min": 0.38},
        })
        self._write("threshold_schedule", {
            "schema_version": "15.4-threshold-schedule-v1",
            "allowed_config_diff_paths": ["detector.intensity_min"],
        })
        self._write("release_gate", {"schema_version": "15.4-release-gate-v1"})
        self._write("formal_authorization", {
            "schema_version": "15.4-formal-run-authorization-v1",
            "formal_comparison_commit": commit,
            "data_identity": {
                "diagnostic_manifest_sha256": diagnostic_hash,
                "formal_manifest_sha256": formal_hash,
                "delta_22_identity_sha256": delta_hash,
            },
        })
        self._write("t0_replay_gate", {
            "schema_version": "15.4-t0-replay-gate-v1",
            "replay_25": {"status": "PASS"},
            "replay_100": {"status": "PASS"},
        })
        for artifact_id in (
            "t0_25_replay", "t0_100_replay", "t0_formal_report",
            "t1_formal_report", "t2_formal_report", "t_off_formal_report",
        ):
            self._write(artifact_id, {"schema_version": artifact_id})
        self._write("matrix_identity_audit", {
            "schema_version": "15.4-formal-matrix-identity-audit-v1",
            "formal_comparison_commit": commit,
            "source_point_monotonicity": {"status": "PASS"},
        })
        metrics = {"0.50": {"tp": 1}, "0.25": {"tp": 2}}
        self._write("formal_experiment", {
            "schema_version": "15.4-intensity-filter-ablation-v1",
            "formal_comparison_commit": commit,
            "source_point_monotonicity": "PASS",
            "variant_scalar_summaries": {
                name: {"metrics_by_iou": metrics}
                for name in ("T0", "T1", "T2", "T_off")
            },
            "release_gate_results": {
                "T1": {
                    "experiment_valid": True,
                    "release_candidate_qualified": False,
                    "gate_results": {"Gate A": {"passed": False}},
                },
                "T2": {
                    "experiment_valid": True,
                    "release_candidate_qualified": False,
                    "gate_results": {"Gate A": {"passed": True}},
                },
                "T_off": {
                    "experiment_valid": True,
                    "release_candidate_qualified": False,
                    "gate_results": {},
                },
            },
            "qualified_release_candidates": [],
            "selected_release_candidate": None,
            "release_goal_achieved": False,
            "safe_configuration": "T0",
        })

    def _write(self, artifact_id, value):
        path = self.root / self.paths[artifact_id]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
