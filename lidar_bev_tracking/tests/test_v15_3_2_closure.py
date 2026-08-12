import json
from pathlib import Path
import unittest


CLOSURE_PATH = Path("docs/v15_3_2_closure.json")


class V1532ClosureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CLOSURE_PATH.open("r", encoding="utf-8") as stream:
            cls.closure = json.load(stream)

    def test_archive_is_closed_and_complete(self):
        self.assertEqual(self.closure["schema_version"], "15.3.2-closure-archive-v1")
        self.assertEqual(self.closure["status"], "CLOSED")
        self.assertEqual(self.closure["archive_status"], "COMPLETE")

    def test_code_identity_is_clean_and_committed(self):
        identity = self.closure["code_identity"]
        self.assertEqual(identity["branch"], "fix/v15-3-2-o2-point-identity")
        self.assertEqual(len(identity["commit"]), 40)
        self.assertFalse(identity["git_dirty"])
        self.assertFalse(identity["patch_artifact_required"])

    def test_manifest_and_eligible_gt_identity_are_frozen(self):
        identity = self.closure["data_identity"]
        self.assertEqual(identity["num_frames"], 25)
        self.assertEqual(len(identity["ordered_frame_ids"]), 25)
        self.assertEqual(len(set(identity["ordered_frame_ids"])), 25)
        self.assertEqual(identity["oracle_eligible_gt_count"], 95)
        hashes = identity["eligible_gt_key_set_sha256"]
        self.assertEqual(set(hashes), {"C0", "C1", "C2", "C3"})
        self.assertEqual(len(set(hashes.values())), 1)
        self.assertTrue(identity["eligible_gt_key_sets_equal"])
        self.assertTrue(identity["does_not_redefine_positive_gt"])

    def test_final_artifact_supersedes_coordinate_dedup(self):
        artifact = self.closure["source_artifacts"]["point_retention_final"]
        self.assertEqual(
            artifact["sha256"],
            "28e292150fc7872e1180d07d74c1eb1811172daaf8f2193541e1ba64d1b370e6",
        )
        self.assertTrue(artifact["canonical"])
        self.assertTrue(artifact["supersedes_coordinate_dedup_result"])

    def test_o2_contract_uses_source_point_identity(self):
        contract = self.closure["o2_fragment_union_contract"]
        self.assertEqual(contract["point_identity"], "raw_lidar_point_index")
        self.assertEqual(contract["union_operator"], "unique_source_index")
        self.assertFalse(contract["coordinate_row_dedup"])
        self.assertEqual(contract["source_alignment_gate"], "required")
        self.assertTrue(contract["o2_point_identity_fix_applied"])

    def test_o2_fix_does_not_change_material_results(self):
        audit = self.closure["o2_fix_audit"]
        self.assertEqual(audit["unchanged_o2_iou_count"], 21)
        self.assertEqual(audit["o2_minus_o1_material_gain"]["before"], 0)
        self.assertEqual(audit["o2_minus_o1_material_gain"]["after"], 0)
        self.assertEqual(audit["o3_minus_o2_gt_clipped_material_gain"]["before"], 4)
        self.assertEqual(audit["o3_minus_o2_gt_clipped_material_gain"]["after"], 4)
        self.assertTrue(audit["original_mixed_samples_unchanged"])
        self.assertFalse(audit["root_cause_changed_after_o2_fix"])

    def test_core_evidence_and_root_cause_counts_are_conserved(self):
        material = self.closure["material_gain_results"]
        self.assertEqual(material["O4_minus_O3"]["count"], 20)
        self.assertTrue(material["O4_minus_O3"]["core_evidence"])
        causes = self.closure["root_cause_attribution"]
        self.assertEqual(causes["INTENSITY_FILTER_LIMITED"], 13)
        self.assertEqual(causes["MIXED"], 7)
        self.assertEqual(causes["Z_FILTER_LIMITED"], 1)
        self.assertEqual(causes["RAW_GEOMETRY_OBSERVABILITY_LIMITED"], 1)
        self.assertEqual(13 + 7 + 1 + 1, causes["conservation_total"])

    def test_pipeline_is_frozen_and_all_gates_pass(self):
        self.assertFalse(any(self.closure["formal_pipeline_freeze"].values()))
        self.assertTrue(all(
            status == "PASS" for status in self.closure["validation_gates"].values()
        ))

    def test_next_stage_references_frozen_provenance(self):
        next_stage = self.closure["next_stage"]
        self.assertEqual(next_stage["name"], "15.4 Intensity Filter Policy Ablation")
        self.assertEqual(
            next_stage["source_diagnostic_sha256"],
            self.closure["source_artifacts"]["point_retention_final"]["sha256"],
        )
        self.assertTrue(next_stage["may_reference_but_not_overwrite_15_3_2"])


if __name__ == "__main__":
    unittest.main()
