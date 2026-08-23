import unittest
from types import SimpleNamespace

import numpy as np

from bev_tracking.fragment_learning_neighbor_context import (
    evaluate_neighbor_signal,
    nearest_fragment_relation,
)


def fragment(identity, source_indices, fragment_type="STRUCTURED"):
    return {
        "runtime_id": identity,
        "source_indices": np.asarray(source_indices, dtype=np.int64),
        "fragment_type": fragment_type,
        "point_count": len(source_indices),
    }


class FragmentLearningNeighborContextTest(unittest.TestCase):
    def test_nearest_member_point_relation_uses_oriented_seed_frame(self):
        raw = np.asarray([
            [0.0, 0.0, 0.0, 0.2], [0.1, 0.0, 0.0, 0.2],
            [1.0, 0.2, 0.0, 0.2], [3.0, 0.0, 0.0, 0.2],
        ])
        target = fragment(0, [0, 1])
        fragments = {0: target, 2: fragment(2, [2]), 3: fragment(3, [3])}
        geometry = SimpleNamespace(
            major=np.asarray([1.0, 0.0]), minor=np.asarray([0.0, 1.0])
        )
        component = SimpleNamespace(geometry=geometry)
        relation = {
            "best_continuation_axis": "SEED_MAJOR", "seed_outward_side": "POS"
        }
        result = nearest_fragment_relation(raw, target, fragments, component, relation)
        self.assertEqual(result["nearest_neighbor_fragment_identity"], 2)
        self.assertAlmostEqual(result["nearest_neighbor_distance"], np.hypot(0.9, 0.2))
        self.assertAlmostEqual(result["nearest_neighbor_outward_projection"], 0.9)
        self.assertAlmostEqual(result["nearest_neighbor_lateral_projection"], 0.2)

    def test_negative_seed_side_reverses_outward_projection(self):
        raw = np.asarray([[0.0, 0.0, 0.0, 0.2], [1.0, 0.0, 0.0, 0.2]])
        target = fragment(0, [0], "SINGLETON")
        geometry = SimpleNamespace(
            major=np.asarray([1.0, 0.0]), minor=np.asarray([0.0, 1.0])
        )
        result = nearest_fragment_relation(
            raw, target, {0: target, 1: fragment(1, [1], "SINGLETON")},
            SimpleNamespace(geometry=geometry),
            {"best_continuation_axis": "SEED_MAJOR", "seed_outward_side": "NEG"},
        )
        self.assertAlmostEqual(result["nearest_neighbor_outward_projection"], -1.0)

    def test_signal_requires_overall_fold1_fold5_and_four_fold_consistency(self):
        records = []
        for fold in range(1, 6):
            for frame_offset in range(3):
                frame = f"{fold}-{frame_offset}"
                for label, value in (("POSITIVE", 4.0), ("N1", 1.0)):
                    row = {
                        "label": label, "fold": fold, "frame_id": frame,
                        "fragment_type": "STRUCTURED",
                    }
                    for field in (
                        "nearest_neighbor_distance",
                        "nearest_neighbor_outward_projection",
                        "nearest_neighbor_lateral_projection",
                        "nearest_neighbor_absolute_lateral_projection",
                        "nearest_neighbor_outward_alignment_cosine",
                    ):
                        row[field] = (
                            value + 0.1 * frame_offset
                            if field == "nearest_neighbor_outward_projection"
                            else 1.0
                        )
                    records.append(row)
        result = evaluate_neighbor_signal(records)
        self.assertEqual(result["LOCAL_FRAGMENT_SUPPORT_COMPLEMENTARITY"], "SUPPORTED")
        self.assertEqual(result["NEIGHBORING_FRAGMENT_CONTEXT_SIGNAL"], "SUPPORTED")


if __name__ == "__main__":
    unittest.main()
