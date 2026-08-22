import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bev_tracking.fragment_learning_dataset import (
    DatasetConstructionError,
    DatasetLabelConflict,
    MODEL_FEATURE_FIELDS,
    build_dataset_row,
    build_offline_fragment_label,
    extract_runtime_fragment_features,
    run_feature_leakage_gate,
    select_fragment_learning_manifest,
    write_fragment_learning_manifest,
)
from bev_tracking.fragment_phase0 import build_candidate_fragments
from bev_tracking.gesr_v1 import build_seed_components_optimized


class FragmentLearningDatasetDay1Test(unittest.TestCase):
    def _runtime_inputs(self):
        seed = np.asarray(
            [
                [0.0, -0.1, 0.0, 0.5],
                [0.0, 0.1, 0.0, 0.5],
                [0.3, -0.1, 0.0, 0.5],
                [0.3, 0.1, 0.0, 0.5],
            ],
            dtype=np.float64,
        )
        components = [
            item for item in build_seed_components_optimized(seed, [1, 2, 3, 4])
            if item.geometry.valid
        ]
        candidates = np.asarray(
            [[1.0, 0.0, 0.0, 0.20], [1.2, 0.0, 0.0, 0.30]],
            dtype=np.float64,
        )
        fragment = build_candidate_fragments(candidates, [10, 11])[0]
        return fragment, components

    def test_manifest_selection_is_frozen_excludes_fixed100_and_reads_no_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("velodyne", "label_2", "calib"):
                (root / "training" / folder).mkdir(parents=True)
            for value in range(70):
                frame = f"{value:06d}"
                (root / "training" / "velodyne" / f"{frame}.bin").touch()
                (root / "training" / "label_2" / f"{frame}.txt").touch()
                (root / "training" / "calib" / f"{frame}.txt").touch()
            fixed = root / "fixed.txt"
            fixed.write_text("000000\n000001\n", encoding="utf-8")
            first = select_fragment_learning_manifest(root, fixed)
            second = select_fragment_learning_manifest(root, fixed)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 64)
            self.assertEqual(first, sorted(first))
            self.assertFalse({"000000", "000001"} & set(first))
            output = write_fragment_learning_manifest(first, root / "manifest.txt")
            self.assertEqual(len(output.read_text().splitlines()), 64)

    def test_manifest_fails_instead_of_replacing_when_insufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("velodyne", "label_2", "calib"):
                (root / "training" / folder).mkdir(parents=True)
            fixed = root / "fixed.txt"
            fixed.write_text("", encoding="utf-8")
            with self.assertRaises(DatasetConstructionError):
                select_fragment_learning_manifest(root, fixed)

    def test_runtime_features_are_finite_and_GT_free(self):
        fragment, components = self._runtime_inputs()
        features = extract_runtime_fragment_features(fragment, components)
        self.assertEqual(tuple(features), MODEL_FEATURE_FIELDS)
        self.assertNotIn("gt_box", inspect.signature(extract_runtime_fragment_features).parameters)
        self.assertEqual(features["fragment_axis_valid"], 1.0)
        self.assertEqual(features["seed_relation_valid"], 1.0)
        numeric = [value for key, value in features.items() if key != "fragment_type"]
        self.assertTrue(np.isfinite(numeric).all())

    def test_singleton_and_no_relation_use_flagged_zero_sentinels(self):
        fragment = build_candidate_fragments(
            np.asarray([[1.0, 0.0, 0.0, 0.2]], dtype=np.float64), [10]
        )[0]
        features = extract_runtime_fragment_features(fragment, [])
        self.assertEqual(features["fragment_axis_valid"], 0.0)
        self.assertEqual(features["seed_relation_valid"], 0.0)
        self.assertEqual(features["orientation_difference"], 0.0)
        self.assertEqual(features["endpoint_gap"], 0.0)

    def test_positive_is_existential_even_when_another_GT_is_not_evaluable(self):
        record = build_offline_fragment_label(
            [{"gt_id": "gt_1", "delta_iou": 0.11}, {"gt_id": "gt_2", "delta_iou": None}],
            strict_background=False,
        )
        self.assertEqual(record["label"], "POSITIVE")

    def test_N1_requires_all_associated_GT_to_be_evaluable(self):
        n1 = build_offline_fragment_label(
            [{"gt_id": "gt_1", "delta_iou": 0.02}, {"gt_id": "gt_2", "delta_iou": 0.09}],
            strict_background=False,
        )
        incomplete = build_offline_fragment_label(
            [{"gt_id": "gt_1", "delta_iou": 0.02}, {"gt_id": "gt_2", "delta_iou": None}],
            strict_background=False,
        )
        self.assertEqual(n1["label"], "N1")
        self.assertEqual(incomplete["label"], "UNLABELED_OTHER")
        self.assertEqual(incomplete["label_reason"], "LABEL_NOT_FULLY_EVALUABLE")

    def test_strict_background_conflict_is_terminal(self):
        with self.assertRaises(DatasetLabelConflict):
            build_offline_fragment_label(
                [], strict_background=True,
                positive_or_neutral_annotation_association=True,
            )
        self.assertEqual(
            build_offline_fragment_label([], strict_background=True)["label"], "N0"
        )

    def test_row_isolation_and_leakage_gate(self):
        fragment, components = self._runtime_inputs()
        features = extract_runtime_fragment_features(fragment, components)
        label = build_offline_fragment_label(
            [{"gt_id": "gt_1", "delta_iou": 0.12}], strict_background=False
        )
        row = build_dataset_row(
            frame_id="1", canonical_fragment_identity=10,
            model_features=features, label_record=label,
            diagnostics={"gt_id": "gt_1", "counterfactual_iou": 0.4},
        )
        result = run_feature_leakage_gate([row], MODEL_FEATURE_FIELDS)
        self.assertEqual(result["result"], "PASS")
        self.assertNotIn("gt_id", row["model_feature_fields"])
        changed_diagnostics = dict(row)
        changed_diagnostics["diagnostic_fields"] = {"gt_id": "changed"}
        self.assertEqual(
            row["model_feature_fields"], changed_diagnostics["model_feature_fields"]
        )
        self.assertEqual(
            run_feature_leakage_gate([row], [*MODEL_FEATURE_FIELDS, "label"])["result"],
            "FAIL",
        )


if __name__ == "__main__":
    unittest.main()
