import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/00_shared/models"))


class ModelContractTest(unittest.TestCase):
    def test_training_runner_uses_public_model_utilities(self):
        import run_nested_hierarchical_pair_anchor_v25 as runner
        import training_utils

        self.assertIs(runner.metrics, training_utils.metrics)
        self.assertFalse(hasattr(training_utils, "PowerEncoder"))

    def test_forward_probability_contract(self):
        import torch
        from camber import build_camber

        generator = torch.Generator().manual_seed(20261105)
        eeg = torch.randn(6, 64, 1000, generator=generator)
        meg = torch.randn(6, 64, 1000, generator=generator)
        model = build_camber(dropout=0.0)
        model.fit_controller_calibration(eeg, meg)
        model.fit_anchor_mixing_calibration(eeg, meg)
        model.eval()
        with torch.inference_mode():
            detail = model.forward_with_details(eeg, meg)
        probability = detail["log_probability"].exp()
        self.assertEqual(tuple(probability.shape), (6, 3))
        torch.testing.assert_close(probability.sum(dim=1), torch.ones(6))
        self.assertEqual(tuple(detail["temporal_weights"].shape), (6, 8, 4))
        self.assertEqual(tuple(detail["branch_weights"].shape), (6, 4))


if __name__ == "__main__":
    unittest.main()
