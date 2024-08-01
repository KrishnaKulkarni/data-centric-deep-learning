import torch
import numpy as np
from typing import List

from .utils import fix_random_seed
from sklearn.cluster import KMeans

import random

import pdb

def random_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Randomly pick examples.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  fix_random_seed(42)
  
  # ================================

  indices = list(range(len(pred_probs)))
  random.shuffle(indices)
  indices = indices[:budget]

  return indices

def uncertainty_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples where the model is the least confident in its predictions.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''

  # ===============================
  # NOTE: I misunderstood the definition of uncertainty sampling:
  # It does not mean: choose examples where the model is least sure that its chosen class / top class is correct
  # It instead means: choose examples where the model is model would be most uncertain of choice, regardless of whether it
  # chose the top class or any of the other classes (i.e. model is maximally ambivalent among across all of its options)
  # Approach 1
  # build a heap of probs
  # for each row: 
    # find its max probability prediction
    # if the len(heap) is < budget
    #.   insert [index, max_pred_prob] to heap
    # else if that max_pred_prob is too low (lower than the highest value of our heap)
      # pop from heap (to make space)
      # insert [index, max_pred_prob] to heap
    # returns the heap, mapped to its elements' indices (perhaps in order)

  # Approach 2
  # map pred_probs to a list of (index, max_pred_prob) pairs
  # sort the list by max_pred_prob ascending
  # take first B (e.g. first 1000)
  # map those to their indices
  max_probs, _ = torch.max(pred_probs, dim=1)
  _, sorted_indices = torch.sort(max_probs, descending=False)
  indices = sorted_indices[:budget].tolist()

  return indices

def margin_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples where the difference between the top two predicted probabilities is the smallest.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''  
  # ================================

  # Find the top two predicted probabilities for each example
  top2_probs, _ = torch.topk(pred_probs, 2, dim=1)

  # Calculate the margin (difference between the top two probabilities)
  margins = top2_probs[:, 0] - top2_probs[:, 1]

  # Get the indices of the examples with the smallest margins
  _, sorted_indices = torch.sort(margins)

  # Select the first 'budget' number of indices
  indices = sorted_indices[:budget].tolist()

  return indices

def entropy_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples with the highest entropy in the predicted probabilities.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  epsilon = 1e-6
  # ================================
  # FILL ME OUT
  # Entropy is defined as -E_classes[log p(class | input)] aja the expected log probability
  # over all K classes. See https://en.wikipedia.org/wiki/Entropy_(information_theory).
  # Sort the indices by the entropy of the predicted probabilities from high to low.
  # Take the first 1000.
  # HINT: Add epsilon when taking a log for entropy computation
  # ================================
  return indices
