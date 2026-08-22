import csv
import json
import tempfile
import unittest
from pathlib import Path

from bev_tracking.fragment_learning_split import (
    FragmentLearningSplitError,
    generate_fragment_learning_splits,
)


class FragmentLearningSplitTest(unittest.TestCase):
    def _write_dataset(self, root, *, sufficient=True):
        root = Path(root)
        frames = [f"{value:06d}" for value in range(64)]
        (root / "fragment_learning_dev_manifest.txt").write_text(
            "\n".join(frames) + "\n", encoding="utf-8"
        )
        summary = {
            "DATASET_SUFFICIENCY": {"result": "PASS" if sufficient else "FAIL"},
            "FEATURE_LEAKAGE_GATE": {"result": "PASS"},
            "MODEL_TRAINING_PERFORMED": False,
            "split_generated": False,
        }
        (root / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        training_rows = []
        with (root / "fragment_dataset.jsonl").open("w", encoding="utf-8") as handle:
            for frame_index, frame_id in enumerate(frames):
                for label_index, label in enumerate(("POSITIVE", "N1", "N0")):
                    identity = frame_index * 10 + label_index
                    row = {
                        "metadata_fields": {
                            "frame_id": frame_id,
                            "canonical_fragment_identity": identity,
                        },
                        "label_field": {"label": label},
                        "model_feature_fields": {"test_feature": 1.0},
                        "diagnostic_fields": {},
                    }
                    handle.write(json.dumps(row) + "\n")
                    training_rows.append(label)
                handle.write(
                    json.dumps(
                        {
                            "metadata_fields": {
                                "frame_id": frame_id,
                                "canonical_fragment_identity": frame_index * 10 + 9,
                            },
                            "label_field": {"label": "UNLABELED_OTHER"},
                            "model_feature_fields": {"test_feature": 1.0},
                            "diagnostic_fields": {},
                        }
                    )
                    + "\n"
                )
        with (root / "X_model.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["test_feature"])
            writer.writerows([[1.0] for _ in training_rows])
        with (root / "y_labels.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_row", "label"))
            writer.writeheader()
            for index, label in enumerate(training_rows):
                writer.writerow({"sample_row": index, "label": label})
        fixed = root / "fixed.txt"
        fixed.write_text("007000\n", encoding="utf-8")
        return fixed

    def test_generates_deterministic_group_isolated_valid_five_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixed = self._write_dataset(root)
            first, split_path = generate_fragment_learning_splits(root, fixed)
            second, _ = generate_fragment_learning_splits(root, fixed)
            self.assertEqual(first, second)
            self.assertEqual(first["SPLIT_VALIDITY"], "PASS")
            self.assertEqual(len(first["folds"]), 5)
            self.assertEqual(first["training_sample_count"], 192)
            self.assertEqual(len(first["frame_to_validation_fold"]), 64)
            self.assertEqual(len(first["sample_assignments"]), 192)
            self.assertTrue(split_path.exists())
            seen_frames = set()
            for fold in first["folds"]:
                current = set(fold["validation_frame_ids"])
                self.assertFalse(seen_frames & current)
                seen_frames |= current
                self.assertEqual(fold["result"], "PASS")
            self.assertEqual(len(seen_frames), 64)
            updated = json.loads((root / "summary.json").read_text())
            self.assertTrue(updated["split_generated"])
            self.assertEqual(updated["SPLIT_VALIDITY"], "PASS")
            self.assertFalse(updated["MODEL_TRAINING_PERFORMED"])

    def test_refuses_split_when_dataset_sufficiency_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixed = self._write_dataset(root, sufficient=False)
            with self.assertRaises(FragmentLearningSplitError):
                generate_fragment_learning_splits(root, fixed)
            self.assertFalse((root / "fragment_learning_splits.json").exists())


if __name__ == "__main__":
    unittest.main()
