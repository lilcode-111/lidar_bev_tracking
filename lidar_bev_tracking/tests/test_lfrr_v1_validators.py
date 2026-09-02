import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import bev_tracking.lfrr_v1_validators as validators
from bev_tracking.fragment_learning_training import M1_PARAMETERS


class LFRRV1ValidatorsTest(unittest.TestCase):
    def test_m2_alignment_is_strict_and_does_not_retrain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments = [{
                "sample_row": index,
                "frame_id": f"{index // 100:06d}",
                "canonical_fragment_identity": index,
                "label": "POSITIVE" if index < 56 else ("N1" if index < 177 else "N0"),
                "validation_fold": index % 5 + 1,
            } for index in range(3252)]
            (root / "fragment_learning_splits.json").write_text(json.dumps({
                "sample_assignments": assignments
            }))
            fields = (
                "sample_row", "frame_id", "canonical_fragment_identity", "label",
                "validation_fold", "M2_score", "M2_prediction_at_0_50",
            )
            with (root / "lightgbm_v2_oof_predictions.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in assignments:
                    score = 0.75 if row["label"] == "POSITIVE" else 0.25
                    writer.writerow({**row, "M2_score": score, "M2_prediction_at_0_50": int(score >= 0.5)})
            (root / "lightgbm_v2_development_result.json").write_text(json.dumps({
                "M2_parameters": M1_PARAMETERS,
                "N0_FP": {"M2": 225},
                "M2": {"FP": 252},
                "P_vs_N1": {"M2_AP": 0.5402350133580981},
            }))
            result = validators.validate_m2_baseline_alignment(root)
            self.assertEqual(result["result"], "PASS")
            with (root / "lightgbm_v2_oof_predictions.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["validation_fold"] = "5"
            with (root / "lightgbm_v2_oof_predictions.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(validators.validate_m2_baseline_alignment(root)["result"], "FAIL")

    def test_normalization_validator_detects_validation_isolation(self):
        catalog = {
            ("000001", 1): {"base24": [1.0] * 24},
            ("000001", 2): {"base24": [3.0] * 24},
            ("000002", 3): {"base24": [999.0] * 24},
        }
        targets = [
            {"sample_row": 0, "frame_id": "000001", "canonical_fragment_identity": 1,
             "validation_fold": 1, "target_margin": 2.0,
             "neighbor_relations": [{"fragment_identity": 2, "delta_u": 1.0, "delta_v": 2.0}]},
            {"sample_row": 1, "frame_id": "000002", "canonical_fragment_identity": 3,
             "validation_fold": 2, "target_margin": 999.0,
             "neighbor_relations": []},
        ]
        normal = validators.FoldNormalizer.fit(catalog, targets, 2)
        changed = json.loads(json.dumps(targets))
        changed[1]["target_margin"] = -123456.0
        other = validators.FoldNormalizer.fit(catalog, changed, 2)
        self.assertEqual(normal.margin_mean, other.margin_mean)
        self.assertTrue(np.array_equal(normal.base_mean, other.base_mean))

    @unittest.skipIf(validators.torch is None, "PyTorch CPU package not installed locally")
    def test_frozen_model_has_no_optimizer_and_exact_parameter_count(self):
        model = validators._model_for_validation()
        self.assertEqual(sum(value.numel() for value in model.parameters()), 2353)
        self.assertFalse(hasattr(model, "optimizer"))


if __name__ == "__main__":
    unittest.main()
