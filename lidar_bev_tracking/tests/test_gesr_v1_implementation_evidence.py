import hashlib
import json
from pathlib import Path
import unittest


class GESRV1ImplementationEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.path = cls.root / "docs/v15_5_gesr_v1_implementation_evidence.json"
        cls.evidence = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_all_bound_source_artifact_hashes_resolve(self):
        for record in self.evidence["source_artifacts"]["files"]:
            path = self.root / record["path"]
            self.assertTrue(path.is_file(), record["path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, record["sha256"], record["path"])

    def test_implementation_is_complete_but_formal_execution_is_not_authorized(self):
        self.assertEqual(
            self.evidence["status"],
            "IMPLEMENTATION_COMPLETE_AWAITING_PHASE2_AUTHORIZATION",
        )
        authorization = self.evidence["authorization"]
        self.assertEqual(authorization["ALGORITHM_IMPLEMENTATION"], "AUTHORIZED")
        self.assertEqual(authorization["PHASE2_FORMAL_EXECUTION"], "NOT_AUTHORIZED")
        self.assertEqual(authorization["PHASE3_FORMAL_EXECUTION"], "NOT_AUTHORIZED")

    def test_test_summary_and_compliance_are_closed(self):
        summary = self.evidence["test_summary"]
        self.assertEqual(summary["test_count"], 50)
        self.assertEqual(summary["pass_count"], 50)
        self.assertEqual(summary["fail_count"], 0)
        self.assertFalse(summary["formal_dataset_used"])
        compliance = self.evidence["compliance"]
        self.assertTrue(all(value is False or value == 0 for value in compliance.values()))

    def test_reference_optimized_equivalence_is_complete(self):
        self.assertTrue(all(self.evidence["equivalence_scope"].values()))


if __name__ == "__main__":
    unittest.main()
