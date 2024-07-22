import torch
import numpy as np
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression

def get_ks_score(tr_probs, te_probs):
  score = None
  # ============================
  # Compute the p-value from the Kolmogorov-Smirnov test.
  # You may use the imported `ks_2samp`.
  # 
  # Type:
  # --
  # tr_probs: torch.Tensor
  #   predicted probabilities from training test
  # te_probs: torch.Tensor
  #   predicted probabilities from test test
  # score: float - between 0 and 1
  # ============================

  # Q: Do we actually need to convert tr_prob/te_prob to numpy
  ksTestResult = ks_2samp(tr_probs.numpy(), te_probs.numpy())

  score = ksTestResult.pvalue

  return score

def get_hist_score(tr_probs, te_probs, bins=10):
  # ============================
  # Compute histogram intersection score. 
  # 
  # Type:
  # --
  # tr_probs: torch.Tensor
  #   predicted probabilities from training test
  # te_probs: torch.Tensor
  #   predicted probabilities from test test
  # score: float - between 0 and 1
  # 
  # ============================

  tr_heights, bin_edges = np.histogram(tr_probs, bins=bins, density=True)
  te_heights, _ = np.histogram(te_probs, bins=bins, density=True)

  score = 0
  for i in range(len(bin_edges) - 1):
    intersection_height = min(tr_heights[i], te_heights[i])
    bin_width = bin_edges[i + 1] - bin_edges[i]
    intersection_area = intersection_height * bin_width

    score = score + intersection_area
  
  return score

def get_vocab_outlier(tr_vocab, te_vocab):
  score = None
  # ============================
  # 
  # Compute the percentage of the test vocabulary
  # that does not appear in the training vocabulary. A score
  # of 0 would mean all of the words in the test vocab
  # appear in the training vocab. A score of 1 would mean
  # none of the new words have been seen before. 
  # 
  # Type:
  # --
  # tr_vocab: dict[str, int]
  #   Map from word to count for training examples
  # te_vocab: dict[str, int]
  #   Map from word to count for test examples
  # score: float (between 0 and 1)
  # ============================
  shared_vocab_count = len(tr_vocab.keys() & te_vocab.keys())
  if len(te_vocab) > 0:
    score = 1 - (shared_vocab_count / len(te_vocab))
  else:
    score = 1

  return score

class MonitoringSystem:

  def __init__(self, tr_vocab, tr_probs, tr_labels):
    self.tr_vocab = tr_vocab
    self.tr_probs = tr_probs
    self.tr_labels = tr_labels

  def calibrate(self, tr_probs, tr_labels, te_probs):
    tr_probs_cal = None
    te_probs_cal = None
    # ============================
    # Calibrate probabilities with isotonic regression using 
    # the training probabilities and labels. 
    # 
    # Pseudocode:
    # --
    # use IsotonicRegression(out_of_bounds='clip')
    #   See documentation for `out_of_bounds` description.
    # tr_probs_cal = fit calibration model
    # te_probs_cal = evaluate using fitted model
    # 
    # Type:
    # --
    # `tr_probs_cal`: torch.Tensor. Note that sklearn
    # returns a NumPy array. You will need to cast 
    # it to a torch.Tensor.
    # 
    # `te_probs_cal`: torch.Tensor
    # ============================

    # tr_probs = tr_probs.numpy()
    # te_probs = te_probs.numpy()
    # tr_labels = tr_labels.numpy()

    iso_reg_model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso_reg_model.fit(tr_probs.numpy(), tr_labels.numpy())

    # tr_probs_cal = iso_reg_model.transform(tr_probs)
    # te_probs_cal = iso_reg_model.transform(te_probs)
    tr_probs_cal = torch.tensor(iso_reg_model.predict(tr_probs.numpy()), dtype=tr_probs.dtype)
    te_probs_cal = torch.tensor(iso_reg_model.predict(te_probs.numpy()), dtype=te_probs.dtype)

    return tr_probs_cal, te_probs_cal

  def monitor(self, te_vocab, te_probs):
    tr_probs, te_probs = self.calibrate(self.tr_probs, self.tr_labels, te_probs)

    # compute metrics. 
    ks_score = get_ks_score(tr_probs, te_probs)
    hist_score = get_hist_score(tr_probs, te_probs)
    outlier_score = get_vocab_outlier(self.tr_vocab, te_vocab)

    metrics = {
      'ks_score': ks_score,
      'hist_score': hist_score,
      'outlier_score': outlier_score,
    }
    return metrics
