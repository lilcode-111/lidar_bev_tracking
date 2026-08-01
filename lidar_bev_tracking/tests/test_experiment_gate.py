import copy
import tempfile
import unittest
from pathlib import Path

from bev_tracking.experiment_gate import (
    ExperimentGateError,
    load_a0_prime_declaration,
    sha256_file,
    validate_a0_prime_declaration,
)


class ExperimentGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.declaration_path = cls.repo_root / "configs" / "experiments" / "v15" / "a0_prime_geometry_corrected.json"

    def test_frozen_a0_prime_declaration_and_manifest_hashes_are_valid(self):
        declaration = load_a0_prime_declaration(self.declaration_path, repo_root=self.repo_root)

        self.assertEqual(declaration["baseline_id"], "A0_prime_geometry_corrected")
        self.assertEqual(declaration["source_run"]["num_frames"], 100)
        self.assertEqual(declaration["metrics_by_iou"]["0.50"]["tp"], 4)
        self.assertEqual(declaration["metrics_by_iou"]["0.25"]["tp"], 12)

    def test_baseline_declaration_rejects_changed_intensity(self):
        declaration = load_a0_prime_declaration(self.declaration_path, repo_root=self.repo_root)
        changed = copy.deepcopy(declaration)
        changed["algorithm_config"]["detector"]["intensity_min"] = 0.0

        with self.assertRaises(ExperimentGateError):
            validate_a0_prime_declaration(changed, repo_root=self.repo_root)

    def test_baseline_declaration_rejects_changed_manifest_hash(self):
        declaration = load_a0_prime_declaration(self.declaration_path, repo_root=self.repo_root)
        changed = copy.deepcopy(declaration)
        changed["dataset"]["diagnostic_manifest_sha256"] = "0" * 64

        with self.assertRaises(ExperimentGateError):
            validate_a0_prime_declaration(changed, repo_root=self.repo_root)

    def test_manifest_hash_is_stable_across_lf_and_crlf(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf_path = Path(tmp) / "lf.txt"
            crlf_path = Path(tmp) / "crlf.txt"
            lf_path.write_bytes(b"000001\n000002\n")
            crlf_path.write_bytes(b"000001\r\n000002\r\n")

            self.assertEqual(sha256_file(lf_path), sha256_file(crlf_path))


if __name__ == "__main__":
    unittest.main()
