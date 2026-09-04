import unittest

import numpy as np

import bev_tracking.lfrr_phase3 as phase3


class Phase3Test(unittest.TestCase):
    def test_signal_mapping_priority_is_frozen(self):
        def gates(a="PASS", b="PASS", seed="PASS", fold="PASS"):
            return {
                "LAYER_A_PRESERVATION_GATE": {"result": a},
                "LAYER_B_INCREMENTAL_GAIN_GATE": {"result": b},
                "SEED_STABILITY_GATE": {"result": seed},
                "FOLD_STABILITY_GATE": {"result": fold},
            }
        self.assertEqual(phase3.map_phase3_signal(gates(), False), "NOT_EVALUATED")
        self.assertEqual(phase3.map_phase3_signal(gates(a="FAIL", b="FAIL")), "NOT_SUPPORTED")
        self.assertEqual(phase3.map_phase3_signal(gates(a="FAIL")), "NOT_ACCEPTABLE_LAYER_A_REGRESSION")
        self.assertEqual(phase3.map_phase3_signal(gates(b="FAIL")), "NO_INCREMENTAL_CONTEXT_VALUE")
        self.assertEqual(phase3.map_phase3_signal(gates(seed="FAIL")), "UNSTABLE")
        self.assertEqual(phase3.map_phase3_signal(gates()), "SUPPORTED")

    def test_stable_sigmoid_handles_extreme_raw_margin(self):
        values = phase3._sigmoid(np.asarray([-1000.0, 0.0, 1000.0]))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[1], 0.5)
        self.assertEqual(values[2], 1.0)

    @unittest.skipIf(phase3.torch is None, "PyTorch CPU package not installed locally")
    def test_context_model_contract_and_empty_bypass(self):
        model = phase3.M2AnchoredContextModel()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 277)
        torch = phase3.torch
        empty = model(torch.zeros(4, 0, 26), torch.zeros(4, 0, dtype=torch.bool))
        self.assertTrue(torch.equal(empty, torch.zeros_like(empty)))
        values = torch.randn(3, 2, 26)
        mask = torch.ones(3, 2, dtype=torch.bool)
        self.assertTrue(torch.equal(model(values, mask), torch.zeros(3)))

    @unittest.skipIf(phase3.torch is None, "PyTorch CPU package not installed locally")
    def test_context_is_permutation_invariant(self):
        torch = phase3.torch
        model = phase3.M2AnchoredContextModel()
        # Make the final layer nonzero so the test exercises MEAN pooling, not only initialization.
        with torch.no_grad():
            model.residual_head[-1].weight.fill_(0.25)
        values = torch.randn(2, 3, 26)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
        permutation = torch.tensor([2, 0, 1])
        self.assertTrue(torch.allclose(
            model(values, mask), model(values[:, permutation], mask[:, permutation]),
            atol=1e-7, rtol=1e-7,
        ))


if __name__ == "__main__":
    unittest.main()
