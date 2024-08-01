import unittest
import torch
from fashion.strategies import(
    random_sampling, uncertainty_sampling, margin_sampling, entropy_sampling
)

class TestRandomSampling(unittest.TestCase):
    def setUp(self):
        # Setting up a fixed random seed for reproducibility
        self.seed = 442
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

    def test_sample_case(self):
        pred_probs = torch.Tensor([
            [0.3, 0, 0.05, 0.15, 0.02, 0.18, 0.05, 0.01, 0.09, 0.15],
            [0.1, 0.05, 0.1, 0.15, 0.2, 0.05, 0.05, 0.1, 0.15, 0.05],
            [0.05, 0.1, 0.2, 0.05, 0.1, 0.05, 0.2, 0.05, 0.1, 0.1],
            [0.12, 0.08, 0.1, 0.15, 0.16, 0.08, 0.04, 0.13, 0.1, 0.04],
            [0.09, 0.11, 0.13, 0.07, 0.16, 0.09, 0.1, 0.1, 0.05, 0.1]
        ])
        budget = 2

        actual_indices = random_sampling(pred_probs, budget)
        expected_indices = [3, 1] # Randomly chooses indices for 4th and 2nd example given seed=42
        
        self.assertEqual(actual_indices, expected_indices, "The sample case output indices are incorrect")

class TestUncertaintySampling(unittest.TestCase):
    def setUp(self):
        # Setting up a fixed random seed for reproducibility
        self.seed = 42
        torch.manual_seed(self.seed)
        
        # Generate a tensor of predicted probabilities for a 10-class classification problem
        self.pred_probs = torch.rand((5000, 10))  # Example tensor with 5000 samples and 10 classes

    def test_output_length(self):
        budget = 1000
        indices = uncertainty_sampling(self.pred_probs, budget)
        self.assertEqual(len(indices), budget, "The length of the output should be equal to the budget.")

    def test_output_unique(self):
        budget = 1000
        indices = uncertainty_sampling(self.pred_probs, budget)
        self.assertEqual(len(indices), len(set(indices)), "The output indices should be unique.")

    def test_output_range(self):
        budget = 1000
        indices = uncertainty_sampling(self.pred_probs, budget)
        self.assertTrue(all(0 <= idx < len(self.pred_probs) for idx in indices), "Indices should be within the valid range.")

    def test_sample_case(self):
        pred_probs = torch.Tensor([
            [0.3, 0, 0.05, 0.15, 0.02, 0.18, 0.05, 0.01, 0.09, 0.15], # Class 0, high conf
            [0.1, 0.05, 0.1, 0.15, 0.2, 0.05, 0.05, 0.1, 0.15, 0.05], # Class 4, med conf
            [0.13, 0.09, 0.1, 0.15, 0.14, 0.08, 0.04, 0.13, 0.1, 0.04],# Class 3, low conf
            [0.05, 0.1, 0.2, 0.05, 0.1, 0.05, 0.2, 0.05, 0.1, 0.1],   # Class 2 or 6, med conf
            [0.09, 0.11, 0.13, 0.07, 0.16, 0.09, 0.1, 0.1, 0.05, 0.1] # Class 2, low conf
        ])
        budget = 2

        actual_indices = uncertainty_sampling(pred_probs, budget)
        expected_indices = [2, 4]
        
        self.assertEqual(actual_indices, expected_indices, "The sample case output indices are incorrect")

class TestMarginSampling(unittest.TestCase):
    def setUp(self):
        # Setting up a fixed random seed for reproducibility
        self.seed = 42
        torch.manual_seed(self.seed)
        
        # Generate a tensor of predicted probabilities for a 10-class classification problem
        self.pred_probs = torch.rand((5000, 10))  # Example tensor with 5000 samples and 10 classes

    def test_output_length(self):
        budget = 1000
        indices = margin_sampling(self.pred_probs, budget)
        self.assertEqual(len(indices), budget, "The length of the output should be equal to the budget.")

    def test_output_unique(self):
        budget = 1000
        indices = margin_sampling(self.pred_probs, budget)
        self.assertEqual(len(indices), len(set(indices)), "The output indices should be unique.")

    def test_output_range(self):
        budget = 1000
        indices = margin_sampling(self.pred_probs, budget)
        self.assertTrue(all(0 <= idx < len(self.pred_probs) for idx in indices), "Indices should be within the valid range.")
    
    def test_sample_case(self):
        pred_probs = torch.Tensor([
            [0.3, 0, 0.05, 0.15, 0.02, 0.18, 0.05, 0.01, 0.09, 0.15], # 0.3 - 0.18 = 0.12
            [0.1, 0.05, 0.1, 0.15, 0.2, 0.05, 0.05, 0.1, 0.15, 0.05], # 0.2 - 0.15 = 0.05
            [0.13, 0.09, 0.1, 0.15, 0.14, 0.08, 0.04, 0.13, 0.1, 0.04],# 0.15 - 0.14 = 0.01
            [0.05, 0.1, 0.2, 0.05, 0.1, 0.05, 0.2, 0.05, 0.1, 0.1],   # 0.2 - 0.2 = 0.0
            [0.09, 0.11, 0.13, 0.07, 0.16, 0.09, 0.1, 0.1, 0.05, 0.1] # 0.16 - 0.13 = 0.03
        ])
        budget = 2

        actual_indices = margin_sampling(pred_probs, budget)
        expected_indices = [3, 2]
        
        self.assertEqual(actual_indices, expected_indices, "The sample case output indices are incorrect")

class TestEntropySampling(unittest.TestCase):
    def setUp(self):
        # Setting up a fixed random seed for reproducibility
        self.seed = 42
        torch.manual_seed(self.seed)
        
        # Generate a tensor of predicted probabilities for a 10-class classification problem
        self.pred_probs = torch.rand((5000, 10))  # Example tensor with 5000 samples and 10 classes

    def test_output_length(self):
        budget = 1000
        indices = entropy_sampling(self.pred_probs, budget)
        self.assertEqual(len(indices), budget, "The length of the output should be equal to the budget.")

    def test_output_unique(self):
        budget = 1000
        indices = entropy_sampling(self.pred_probs, budget)
        self.assertEqual(len(indices), len(set(indices)), "The output indices should be unique.")

    def test_output_range(self):
        budget = 1000
        indices = entropy_sampling(self.pred_probs, budget)
        self.assertTrue(all(0 <= idx < len(self.pred_probs) for idx in indices), "Indices should be within the valid range.")
    
    def test_sample_case(self):
        pred_probs = torch.Tensor([
            [0.3, 0, 0.05, 0.15, 0.02, 0.18, 0.05, 0.01, 0.09, 0.15], # H(X) = 2.712
            [0.1, 0.05, 0.1, 0.15, 0.2, 0.05, 0.05, 0.1, 0.15, 0.05], # H(X) = 3.146
            [0.13, 0.09, 0.1, 0.15, 0.14, 0.08, 0.04, 0.13, 0.1, 0.04],# H(X) = 3.213
            [0.05, 0.1, 0.2, 0.05, 0.1, 0.05, 0.2, 0.05, 0.1, 0.1],   # H(X) = 3.122
            [0.09, 0.11, 0.13, 0.07, 0.16, 0.09, 0.1, 0.1, 0.05, 0.1] # H(X) = 3.262
        ])
        budget = 2

        actual_indices = entropy_sampling(pred_probs, budget)
        expected_indices = [4, 2]
        
        self.assertEqual(actual_indices, expected_indices, "The sample case output indices are incorrect")

if __name__ == '__main__':
    unittest.main()
