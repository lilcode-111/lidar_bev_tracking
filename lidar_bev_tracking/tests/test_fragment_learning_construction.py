import csv
import json
import tempfile
import unittest
from pathlib import Path
import zipfile

import numpy as np

from bev_tracking.fragment_learning_construction import (
    ALL_LABELS,
    build_frame_dataset_rows,
    collect_complete_archive_frame_ids,
    construct_fragment_learning_dataset,
    dataset_sufficiency,
    prepare_or_validate_manifest,
    prepare_selected_archive_cache,
)
from bev_tracking.fragment_learning_dataset import (
    DatasetConstructionError,
    MODEL_FEATURE_FIELDS,
)


class FragmentLearningConstructionTest(unittest.TestCase):
    def _write_official_archive_triplet(self, root, frame_ids):
        specs = {
            "data_object_velodyne.zip": ("training/velodyne", ".bin", b"bin"),
            "data_object_label_2.zip": ("training/label_2", ".txt", b"label"),
            "data_object_calib.zip": ("training/calib", ".txt", b"calib"),
        }
        for archive_name, (folder, suffix, payload) in specs.items():
            with zipfile.ZipFile(root / archive_name, "w") as archive:
                for frame_id in frame_ids:
                    archive.writestr(f"{folder}/{frame_id}{suffix}", payload)

    def test_official_archives_select_full_catalog_and_stage_only_64(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_ids = [f"{value:06d}" for value in range(70)]
            self._write_official_archive_triplet(root, frame_ids)
            fixed = root / "fixed.txt"
            fixed.write_text(
                "\n".join(f"{value:06d}" for value in range(6)) + "\n",
                encoding="utf-8",
            )
            manifest = root / "output" / "manifest.txt"
            cache = root / "cache"
            self.assertEqual(len(collect_complete_archive_frame_ids(root)), 70)
            result = prepare_selected_archive_cache(
                root, fixed, manifest, cache
            )
            self.assertEqual(result["complete_archive_frame_count"], 70)
            self.assertEqual(result["selected_frame_count"], 64)
            self.assertEqual(result["newly_extracted_file_count"], 192)
            selected = manifest.read_text().splitlines()
            self.assertEqual(selected, sorted(selected))
            self.assertFalse(set(selected) & {f"{value:06d}" for value in range(6)})
            self.assertEqual(
                len(list((cache / "training" / "velodyne").glob("*.bin"))), 64
            )
            self.assertEqual(
                len(list((cache / "training" / "label_2").glob("*.txt"))), 64
            )
            self.assertEqual(
                len(list((cache / "training" / "calib").glob("*.txt"))), 64
            )
            replay = prepare_selected_archive_cache(
                root, fixed, manifest, cache
            )
            self.assertEqual(replay["newly_extracted_file_count"], 0)

    def test_frame_builder_emits_unique_GT_free_runtime_rows(self):
        raw = np.asarray(
            [
                [5.0, -0.2, 0.0, 0.50],
                [5.0, 0.2, 0.0, 0.50],
                [5.4, -0.2, 0.0, 0.50],
                [5.4, 0.2, 0.0, 0.50],
                [6.0, -0.2, 0.0, 0.20],
                [6.2, 0.2, 0.0, 0.25],
                [20.0, 10.0, 0.0, 0.20],
            ],
            dtype=np.float64,
        )
        boxes = [
            {
                "id": "gt_1", "class_name": "car", "x": 5.5, "y": 0.0,
                "z": 0.0, "length": 4.0, "width": 2.0, "height": 2.0,
                "yaw": 0.0,
            }
        ]
        result = build_frame_dataset_rows(raw, boxes, "1")
        self.assertEqual(result["candidate_point_count"], 3)
        self.assertEqual(result["fragment_count"], 2)
        identities = [
            row["metadata_fields"]["canonical_fragment_identity"]
            for row in result["rows"]
        ]
        self.assertEqual(len(identities), len(set(identities)))
        labels = {row["label_field"]["label"] for row in result["rows"]}
        self.assertIn("N0", labels)
        associated = [
            row for row in result["rows"]
            if row["diagnostic_fields"]["associated_positive_car_GT"]
        ]
        self.assertEqual(len(associated), 1)
        self.assertNotIn("gt_id", associated[0]["model_feature_fields"])

    def test_float32_threshold_candidate_does_not_fake_T2_minus_T0_association(self):
        raw = np.asarray(
            [
                [5.0, -0.2, 0.0, 0.50],
                [5.0, 0.2, 0.0, 0.50],
                [5.4, -0.2, 0.0, 0.50],
                [5.4, 0.2, 0.0, 0.50],
                [6.0, 0.0, 0.0, 0.38],
            ],
            dtype=np.float32,
        )
        box = {
            "id": "gt_1", "class_name": "car", "x": 5.5, "y": 0.0,
            "z": 0.0, "length": 4.0, "width": 2.0, "height": 2.0,
            "yaw": 0.0,
        }
        result = build_frame_dataset_rows(raw, [box], "25")
        boundary_rows = [
            row for row in result["rows"]
            if row["metadata_fields"]["canonical_fragment_identity"] == 4
        ]
        self.assertEqual(len(boundary_rows), 1)
        self.assertEqual(
            boundary_rows[0]["diagnostic_fields"]["associated_positive_car_GT"],
            [],
        )
        self.assertEqual(
            boundary_rows[0]["label_field"]["label"], "UNLABELED_OTHER"
        )

    def test_runtime_features_do_not_change_when_GT_is_removed_or_modified(self):
        raw = np.asarray(
            [
                [5.0, -0.2, 0.0, 0.50],
                [5.0, 0.2, 0.0, 0.50],
                [5.4, -0.2, 0.0, 0.50],
                [5.4, 0.2, 0.0, 0.50],
                [6.0, -0.2, 0.0, 0.20],
                [6.2, 0.2, 0.0, 0.25],
            ],
            dtype=np.float64,
        )
        original = {
            "id": "gt_1", "class_name": "car", "x": 5.5, "y": 0.0,
            "z": 0.0, "length": 4.0, "width": 2.0, "height": 2.0,
            "yaw": 0.0,
        }
        modified = {**original, "x": 30.0, "y": 10.0, "yaw": 1.2}
        variants = [
            build_frame_dataset_rows(raw, [original], "1"),
            build_frame_dataset_rows(raw, [], "1"),
            build_frame_dataset_rows(raw, [modified], "1"),
        ]
        feature_maps = [
            {
                row["metadata_fields"]["canonical_fragment_identity"]:
                    row["model_feature_fields"]
                for row in result["rows"]
            }
            for result in variants
        ]
        self.assertEqual(feature_maps[0], feature_maps[1])
        self.assertEqual(feature_maps[0], feature_maps[2])

    def test_sufficiency_is_frozen_and_does_not_relax_counts(self):
        counts = {label: 0 for label in ALL_LABELS}
        counts.update({"POSITIVE": 30, "N1": 20, "N0": 1})
        support = {label: set() for label in ALL_LABELS}
        support["POSITIVE"] = {f"p{i}" for i in range(10)}
        support["N1"] = {f"n{i}" for i in range(5)}
        support["N0"] = {"b"}
        self.assertEqual(dataset_sufficiency(counts, support)["result"], "PASS")
        counts["POSITIVE"] = 29
        self.assertEqual(dataset_sufficiency(counts, support)["result"], "FAIL")

    def test_streaming_writer_separates_X_labels_and_unlabeled_rows(self):
        def fake_builder(_data_root, frame_id):
            label = "POSITIVE" if int(frame_id) < 10 else "UNLABELED_OTHER"
            features = {
                field: (
                    "STRUCTURED" if field == "fragment_type" else
                    1.0 if field in {"fragment_axis_valid", "seed_relation_valid"} else
                    0.5
                )
                for field in MODEL_FEATURE_FIELDS
            }
            row = {
                "metadata_fields": {
                    "frame_id": frame_id, "canonical_fragment_identity": 1,
                },
                "label_field": {"label": label},
                "model_feature_fields": features,
                "diagnostic_fields": {"label_reason": "test"},
            }
            return {
                "frame_id": frame_id,
                "candidate_point_count": 2,
                "fragment_count": 1,
                "rows": [row],
            }

        frames = [f"{value:06d}" for value in range(64)]
        with tempfile.TemporaryDirectory() as tmp:
            summary = construct_fragment_learning_dataset(
                "unused", frames, tmp, frame_builder=fake_builder
            )
            self.assertEqual(summary["unique_runtime_fragment_count"], 64)
            self.assertEqual(summary["label_counts"]["POSITIVE"], 10)
            self.assertEqual(summary["label_counts"]["UNLABELED_OTHER"], 54)
            self.assertEqual(summary["training_row_count"], 10)
            self.assertEqual(summary["FEATURE_LEAKAGE_GATE"]["result"], "PASS")
            lines = (Path(tmp) / "fragment_dataset.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 64)
            with (Path(tmp) / "X_model.csv").open(newline="") as handle:
                x_rows = list(csv.DictReader(handle))
            with (Path(tmp) / "y_labels.csv").open(newline="") as handle:
                y_rows = list(csv.DictReader(handle))
            self.assertEqual(len(x_rows), 10)
            self.assertEqual(len(y_rows), 10)
            self.assertEqual(tuple(x_rows[0]), MODEL_FEATURE_FIELDS)

    def test_failure_removes_partial_training_tables(self):
        def failing_builder(_data_root, frame_id):
            raise RuntimeError(frame_id)

        frames = [f"{value:06d}" for value in range(64)]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                construct_fragment_learning_dataset(
                    "unused", frames, tmp, frame_builder=failing_builder
                )
            self.assertFalse((Path(tmp) / "fragment_dataset.jsonl.tmp").exists())
            self.assertFalse((Path(tmp) / "X_model.csv.tmp").exists())

    def test_existing_manifest_must_match_frozen_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("velodyne", "label_2", "calib"):
                (root / "training" / folder).mkdir(parents=True)
            for value in range(170):
                frame = f"{value:06d}"
                (root / "training" / "velodyne" / f"{frame}.bin").touch()
                (root / "training" / "label_2" / f"{frame}.txt").touch()
                (root / "training" / "calib" / f"{frame}.txt").touch()
            fixed = root / "fixed.txt"
            fixed.write_text("\n".join(f"{i:06d}" for i in range(100)) + "\n")
            manifest = root / "manifest.txt"
            selected = prepare_or_validate_manifest(root, fixed, manifest)
            self.assertEqual(len(selected), 64)
            manifest.write_text("\n".join(f"{i:06d}" for i in range(64)) + "\n")
            with self.assertRaises(DatasetConstructionError):
                prepare_or_validate_manifest(root, fixed, manifest)


if __name__ == "__main__":
    unittest.main()
