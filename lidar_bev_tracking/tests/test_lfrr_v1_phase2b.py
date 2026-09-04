import unittest

import numpy as np

import bev_tracking.lfrr_v1_phase2b as phase2b


class LFRRV1Phase2BTest(unittest.TestCase):
    @unittest.skipIf(phase2b.torch is None, "PyTorch CPU package not installed locally")
    def test_target_only_model_has_exact_architecture_and_parameter_count(self):
        model = phase2b.TargetOnlyNeuralControl()
        self.assertEqual(sum(value.numel() for value in model.parameters()), 705)
        output = model(phase2b.torch.zeros((3, 25)))
        self.assertEqual(tuple(output.shape), (3,))
        with self.assertRaises(phase2b.Phase2BError):
            model(phase2b.torch.zeros((3, 26)))

    def test_case_a_requires_large_two_layer_recovery(self):
        case, counts = phase2b.classify_preliminary_case(
            0.8, 0.9,
            [0.16] * 5, [0.52] * 5,
            [0.09] * 5, [0.41] * 5,
        )
        self.assertEqual(case, "A")
        self.assertEqual(counts["A_TO_above_FULL"], 5)

    def test_case_b_requires_small_gap_and_below_m2_direction(self):
        case, _ = phase2b.classify_preliminary_case(
            0.1, 0.2,
            [0.10] * 5, [0.43] * 5,
            [0.09] * 5, [0.42] * 5,
        )
        self.assertEqual(case, "B")

    def test_case_c_is_mixed_middle_recovery(self):
        case, _ = phase2b.classify_preliminary_case(
            0.5, 0.6,
            [0.13] * 5, [0.48] * 5,
            [0.09] * 5, [0.42] * 5,
        )
        self.assertEqual(case, "C")

    def test_seed_instability_overrides_case(self):
        case, _ = phase2b.classify_preliminary_case(
            0.8, 0.8,
            [0.10, 0.08, 0.10, 0.08, 0.10],
            [0.45, 0.40, 0.45, 0.40, 0.45],
            [0.09] * 5, [0.42] * 5,
        )
        self.assertEqual(case, "SEED_INSTABILITY")


if __name__ == "__main__":
    unittest.main()
