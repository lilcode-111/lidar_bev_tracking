from pathlib import Path
import tempfile
import unittest

import numpy as np

import bev_tracking.lfrr_v1_training as training


class LFRRV1TrainingTest(unittest.TestCase):
    def _gates(self, a="PASS", b="PASS", fold="PASS", seed="PASS"):
        return {
            "LAYER_A_PRESERVATION_GATE": {"result": a},
            "LAYER_B_IMPROVEMENT_GATE": {"result": b},
            "CROSS_FOLD_STABILITY_GATE": {"result": fold},
            "SEED_STABILITY": {"result": seed},
        }

    def test_development_conclusion_is_mechanical(self):
        self.assertEqual(
            training.map_development_signal(self._gates()), "SUPPORTED"
        )
        self.assertEqual(
            training.map_development_signal(self._gates(seed="FAIL")), "UNSTABLE"
        )
        self.assertEqual(
            training.map_development_signal(self._gates(b="FAIL")), "NOT_SUPPORTED"
        )

    def test_fixed_execution_protocol(self):
        self.assertEqual(training.TRAINING_SEEDS, (15531, 15532, 15533, 15534, 15535))
        self.assertEqual(training.FOLDS, (1, 2, 3, 4, 5))
        self.assertEqual(training.EPOCHS, 100)
        self.assertEqual(training.BATCH_SIZE, 128)
        self.assertEqual(training.LEARNING_RATE, 1.0e-3)
        self.assertEqual(training.WEIGHT_DECAY, 1.0e-4)

    def test_metric_summary_preserves_median_min_max(self):
        rows = []
        for value in (0.2, 0.3, 0.4, 0.5, 0.6):
            base = {
                "average_precision": value, "roc_auc": value,
                "precision_at_0_50": value, "recall_at_0_50": value,
                "f1_at_0_50": value, "FP": int(value * 100),
            }
            rows.append({"metrics": {
                "Overall": dict(base),
                "Layer_A_P_vs_N0": {
                    **base, "N0_FP_at_0_50": int(value * 100),
                    "N0_FPR_at_0_50": value,
                },
                "Layer_B_P_vs_N1": dict(base),
            }})
        summary = training._aggregate_seed_metrics(rows)
        self.assertAlmostEqual(summary["Layer_B_AP"]["median"], 0.4)
        self.assertAlmostEqual(summary["Layer_B_AP"]["min"], 0.2)
        self.assertAlmostEqual(summary["Layer_B_AP"]["max"], 0.6)

    @unittest.skipIf(training.torch is None, "PyTorch CPU package not installed locally")
    def test_one_model_executes_optimizer_and_writes_final_epoch_checkpoint(self):
        torch = training.torch
        original_epochs = training.EPOCHS
        original_batch = training.BATCH_SIZE
        training.EPOCHS = 2
        training.BATCH_SIZE = 4
        try:
            train = {
                "target": torch.randn(8, 25),
                "neighbors": torch.randn(8, 2, 26),
                "neighbor_mask": torch.ones(8, 2, dtype=torch.bool),
                "label": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32),
                "sample_rows": np.arange(8),
            }
            validation = {
                "target": torch.randn(4, 25),
                "neighbors": torch.zeros(4, 1, 26),
                "neighbor_mask": torch.zeros(4, 1, dtype=torch.bool),
                "label": torch.tensor([0, 1, 0, 1], dtype=torch.float32),
                "sample_rows": np.arange(4),
            }
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "model.pt"
                result = training._train_one_model(
                    {"training": train, "validation": validation},
                    15531, 1, checkpoint,
                )
                self.assertTrue(checkpoint.exists())
                self.assertEqual(len(result["training_loss_by_epoch"]), 2)
                self.assertEqual(len(result["validation_probabilities"]), 4)
        finally:
            training.EPOCHS = original_epochs
            training.BATCH_SIZE = original_batch


if __name__ == "__main__":
    unittest.main()
