import unittest
import torch
from fashion.strategies import random_sampling

class TestRandomSampling(unittest.TestCase):
    def setUp(self):
        # Setting up a fixed random seed for reproducibility
        self.seed = 42
        self.pred_probs = torch.rand(5000)  # Example tensor with 5000 elements

    def test_output_length(self):
        budget = 1000
        indices = random_sampling(self.pred_probs, budget)
        self.assertEqual(len(indices), budget, "The length of the output should be equal to the budget.")

    def test_output_unique(self):
        budget = 1000
        indices = random_sampling(self.pred_probs, budget)
        self.assertEqual(len(indices), len(set(indices)), "The output indices should be unique.")

    def test_output_range(self):
        budget = 1000
        indices = random_sampling(self.pred_probs, budget)
        self.assertTrue(all(0 <= idx < len(self.pred_probs) for idx in indices), "Indices should be within the valid range.")

    def test_random_seed_consistency(self):
        budget = 1000
        indices1 = random_sampling(self.pred_probs, budget)
        indices2 = random_sampling(self.pred_probs, budget)
        self.assertEqual(indices1, indices2, "The function should produce the same output for the same random seed.")

    def test_output_for_sample_case(self):
        pred_probs = torch.Tensor([0.3, 0.4, 0.5, 0.6, 0.4, 0.05, 0.8, 0.1, 0, 0.2])
        budget = 3

        actual_indices = random_sampling(pred_probs, budget)
        expected_indices = [7, 3, 2] # Randomly chooses indices for 0.6, 0.5, 0.1 given seed=42
        
        self.assertEqual(actual_indices, expected_indices, "The sample case output indices are incorrect")


if __name__ == '__main__':
    unittest.main()
