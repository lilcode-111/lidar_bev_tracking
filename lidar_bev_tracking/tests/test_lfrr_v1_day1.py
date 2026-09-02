import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import bev_tracking.lfrr_v1 as lfrr
from bev_tracking.lfrr_v1 import (
    EXPECTED_PARAMETER_COUNT,
    FoldNormalizer,
    SmallTargetConditionedDeepSets,
    collate_lfrr_samples,
    construct_frame_representation,
    register_independent_manifest,
)


def catalog_row(identity, seed, x=0.0, y=0.0):
    return {
        "fragment_identity": identity,
        "base24": [float(identity)] + [0.0] * 23,
        "best_seed_component_runtime_id": seed,
        "center_xy": [x, y],
    }


class LFRRV1Day1Test(unittest.TestCase):
    def test_local_set_is_same_seed_sorted_and_keeps_empty_target(self):
        fragments = {
            30: {"runtime_id": 30}, 10: {"runtime_id": 10},
            20: {"runtime_id": 20}, 40: {"runtime_id": 40},
        }
        records = {
            10: catalog_row(10, 7, 0.0, 0.0),
            20: catalog_row(20, 7, 1.0, 2.0),
            30: catalog_row(30, 8, 5.0, 5.0),
            40: catalog_row(40, None, 3.0, 3.0),
        }
        component = SimpleNamespace(
            geometry=SimpleNamespace(
                major=np.asarray([1.0, 0.0]), minor=np.asarray([0.0, 1.0])
            )
        )
        targets = [
            {"sample_row": 0, "frame_id": "000001", "canonical_fragment_identity": 10,
             "label": "POSITIVE", "validation_fold": 1},
            {"sample_row": 1, "frame_id": "000001", "canonical_fragment_identity": 40,
             "label": "N1", "validation_fold": 1},
        ]
        margins = {
            ("000001", 10): {"value": 0.5, "valid": True,
                              "best_seed_component_runtime_id": 7,
                              "second_seed_component_runtime_id": 8},
            ("000001", 40): {"value": None, "valid": False,
                              "best_seed_component_runtime_id": None,
                              "second_seed_component_runtime_id": None},
        }
        with patch.object(lfrr, "_fragment_record", side_effect=lambda item, _: records[item["runtime_id"]]):
            _, output = construct_frame_representation(
                "000001", fragments, {7: component, 8: component}, [], targets, margins
            )
        self.assertEqual([item["fragment_identity"] for item in output[0]["neighbor_relations"]], [20])
        self.assertEqual(output[0]["neighbor_relations"][0]["delta_u"], 1.0)
        self.assertEqual(output[0]["neighbor_relations"][0]["delta_v"], 2.0)
        self.assertEqual(output[1]["neighbor_relations"], [])

    def test_fold_normalizer_uses_training_targets_and_unique_fragments(self):
        catalog = {
            ("000001", 1): {"base24": [1.0] * 24},
            ("000001", 2): {"base24": [3.0] * 24},
            ("000002", 3): {"base24": [1000.0] * 24},
        }
        targets = [
            {"frame_id": "000001", "canonical_fragment_identity": 1,
             "validation_fold": 1, "target_margin": 2.0,
             "neighbor_relations": [
                 {"fragment_identity": 2, "delta_u": 4.0, "delta_v": 8.0},
                 {"fragment_identity": 2, "delta_u": 6.0, "delta_v": 10.0},
             ]},
            {"frame_id": "000002", "canonical_fragment_identity": 3,
             "validation_fold": 2, "target_margin": 999.0,
             "neighbor_relations": []},
        ]
        normalizer = FoldNormalizer.fit(catalog, targets, validation_fold=2)
        self.assertTrue(np.allclose(normalizer.base_mean, 2.0))
        self.assertEqual(normalizer.evidence["unique_base24_fragment_count"], 2)
        self.assertEqual(normalizer.margin_mean, 2.0)
        self.assertTrue(np.allclose(normalizer.delta_mean, [5.0, 9.0]))
        transformed = normalizer.transform_target(catalog[("000001", 1)], None)
        self.assertEqual(transformed.shape, (25,))
        self.assertEqual(transformed[-1], 0.0)

    def test_independent_manifest_is_identity_only_and_excludes_union(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            learning = root / "learning.txt"
            fixed = root / "fixed.txt"
            phase0 = root / "phase0.csv"
            output = root / "independent.txt"
            learning.write_text("\n".join(f"{value:06d}" for value in range(64)) + "\n")
            fixed.write_text("\n".join(f"{value:06d}" for value in range(100, 200)) + "\n")
            phase0.write_text("frame_id\n" + "\n".join(f"{value:06d}" for value in range(60, 69)) + "\n")
            result = register_independent_manifest(
                [f"{value:06d}" for value in range(500)], learning, fixed,
                phase0, output, frame_count=100,
            )
            selected = set(output.read_text().splitlines())
            exclusion = {f"{value:06d}" for value in range(64)} | {
                f"{value:06d}" for value in range(100, 200)
            } | {f"{value:06d}" for value in range(60, 69)}
            self.assertFalse(selected & exclusion)
            self.assertEqual(result["phase0_extra_exclusion_frames"], [
                "000064", "000065", "000066", "000067", "000068"
            ])
            first = output.read_text()
            register_independent_manifest(
                [f"{value:06d}" for value in range(500)], learning, fixed,
                phase0, output, frame_count=100,
            )
            self.assertEqual(output.read_text(), first)

    @unittest.skipIf(lfrr.torch is None, "PyTorch CPU package not installed locally")
    def test_model_identity_masking_empty_set_and_permutation(self):
        model = SmallTargetConditionedDeepSets().eval()
        self.assertEqual(sum(value.numel() for value in model.parameters()), EXPECTED_PARAMETER_COUNT)
        samples = [
            {"sample_row": 0, "label": "N0", "target": np.zeros(25),
             "neighbors": np.empty((0, 26))},
            {"sample_row": 1, "label": "POSITIVE", "target": np.ones(25),
             "neighbors": np.asarray([[1.0] * 26, [2.0] * 26, [3.0] * 26])},
        ]
        batch = collate_lfrr_samples(samples)
        logits, pooled = model(
            batch["target"], batch["neighbors"], batch["neighbor_mask"],
            return_pooled=True,
        )
        self.assertTrue(lfrr.torch.equal(pooled[0], lfrr.torch.zeros(16)))
        order = lfrr.torch.tensor([2, 0, 1])
        permuted_neighbors = batch["neighbors"].clone()
        permuted_neighbors[1] = permuted_neighbors[1, order]
        permuted_mask = batch["neighbor_mask"].clone()
        permuted_mask[1] = permuted_mask[1, order]
        other_logits, other_pooled = model(
            batch["target"], permuted_neighbors, permuted_mask, return_pooled=True
        )
        self.assertTrue(lfrr.torch.allclose(logits, other_logits, atol=1e-6, rtol=1e-6))
        self.assertTrue(lfrr.torch.allclose(pooled, other_pooled, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
