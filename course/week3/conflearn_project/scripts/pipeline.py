import os
import torch
import random
import numpy as np
import pandas as pd
from os.path import join
from pprint import pprint
from torch.utils.data import DataLoader, TensorDataset

from metaflow import FlowSpec, step, Parameter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold

# NOTE: use this library
from cleanlab.filter import find_label_issues

from conflearn.system import ReviewDataModule, SentimentClassifierSystem
from conflearn.utils import load_config, to_json
from conflearn.paths import DATA_DIR, LOG_DIR, CONFIG_DIR


class TrainIdentifyReview(FlowSpec):
  r"""A MetaFlow that trains a sentiment classifier on reviews of luxury beauty
  products using PyTorch Lightning, identifies data quality issues using CleanLab, 
  and prepares them for review in LabelStudio.

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """
  config_path = Parameter('config', help = 'path to config file', default='./train.json')

  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    self.next(self.init_system)

  @step
  def init_system(self):
    r"""Instantiates a data module, pytorch lightning module, 
    and lightning trainer instance.
    """
    # configuration files contain all hyperparameters
    config = load_config(join(CONFIG_DIR, self.config_path))

    # a callback to save best model weights
    checkpoint_callback = ModelCheckpoint(
      dirpath = config.train.ckpt_dir,
      monitor = 'dev_loss',
      mode = 'min',    # look for lowest `dev_loss`
      save_top_k = 1,  # save top 1 checkpoints
      verbose = True,
    )

    trainer = Trainer(
      max_epochs = config.train.optimizer.max_epochs,
      callbacks = [checkpoint_callback],
    )

    # when we save these objects to a `step`, they will be available
    # for use in the next step, through not steps after.
    self.trainer = trainer
    self.config = config

    self.next(self.train_test)

  @step
  def train_test(self):
    """Calls `fit` on the trainer.
    
    We first train and (offline) evaluate the model to see what 
    performance would be without any improvements to data quality.
    """
    # a data module wraps around training, dev, and test datasets
    dm = ReviewDataModule(self.config)

    # a PyTorch Lightning system wraps around model logic
    system = SentimentClassifierSystem(self.config)

    # Call `fit` on the trainer with `system` and `dm`.
    # Our solution is one line.
    self.trainer.fit(system, dm)
    self.trainer.test(system, dm, ckpt_path = 'best')

    # results are saved into the system
    results = system.test_results

    self._print_and_log_results(results= results, log_filename= 'baseline.json')

    self.next(self.crossval)

  def _print_and_log_results(self, results, log_filename):
    pprint(results)

    log_file = join(LOG_DIR, log_filename)
    os.makedirs(LOG_DIR, exist_ok = True)
    to_json(results, log_file)  # save to disk
  
  @step
  def crossval(self):
    r"""Confidence learning requires cross validation to compute 
    out-of-sample probabilities for every element. Each element
    will appear in a single cross validation split exactly once. 
    """
    dm = ReviewDataModule(self.config)
    # combine training and dev datasets
    X = np.concatenate([
      np.asarray(dm.train_dataset.embedding),
      np.asarray(dm.dev_dataset.embedding),
      np.asarray(dm.test_dataset.embedding),
    ])
    y = np.concatenate([
      np.asarray(dm.train_dataset.data.label),
      np.asarray(dm.dev_dataset.data.label),
      np.asarray(dm.test_dataset.data.label),
    ])

    probs = np.zeros(len(X))  # we will fill this in

    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    kf = KFold(n_splits=3)    # create kfold splits

    for train_index, test_index in kf.split(X):
      # ===============================================
      # Types:
      # --
      # predicted_probs: np.array[float] (shape: |test set|)
      # ===============================================
      train_dataloader, test_dataloader = self._create_train_n_test_data_loaders(
        train_index, test_index, X, y
      )
      system = SentimentClassifierSystem(self.config)
      current_fold_probs = self._train_model_and_make_predictions(
        system, train_dataloader, test_dataloader
      )
      probs[test_index] = current_fold_probs

    # Q: Why isn't the following code redundant with the concatenation that happens at the top of the method?
    # create a single dataframe with all input features
    all_df = pd.concat([
      dm.train_dataset.data,
      dm.dev_dataset.data,
      dm.test_dataset.data,
    ])
     # Q: What is `reset_index(drop=True)` doing?
    all_df = all_df.reset_index(drop=True)
    # add out-of-sample probabilities to the dataframe
    # Q: How are we ensuring that the probabilities are corrected mapped to their corresponding input?
    all_df['prob'] = probs

    # save to excel file
    all_df.to_csv(join(DATA_DIR, 'prob.csv'), index=False)

    self.all_df = all_df
    self.next(self.inspect)

  def _create_train_n_test_data_loaders(self, train_index, test_index, X, y):
      train_index, test_index = np.array(train_index), np.array(test_index)

      # Create tensor slices
      X_train, X_test = torch.from_numpy(X[train_index]), torch.from_numpy(X[test_index])
      y_train, y_test = torch.from_numpy(y[train_index]), torch.from_numpy(y[test_index])
      # Q: Why did Amy's example append .float() / .long() ?

      # Create train/test datasets using tensors.
      # Q: What is the TensorDataset API? Why do we need to instantiate a new object vs. just using the numpy arrays?
      # Q: Why does this not resemble the training code of #train_test?
      train_dataset = TensorDataset(X_train, y_train)
      test_dataset = TensorDataset(X_test, y_test)

      # Create train/test data loaders from datasets.
      # Q: Why does this not resemble the training code of #train_test?
      # Q: Why is self.config available in this step, given we DID NOT store it in the immediately prior step?
      # Q: Do we NEED to shuffle the data in the Dataloader? Why or why not? What is the effect of doing so?
      train_dataloader = DataLoader(
        train_dataset,
        batch_size = self.config.train.optimizer.batch_size,
        shuffle = True
      )
      test_dataloader = DataLoader(
        test_dataset,
        batch_size = self.config.train.optimizer.batch_size
      )

      return train_dataloader, test_dataloader
  
  def _train_model_and_make_predictions(self, system, train_dataloader, test_dataloader):
      # Create `Trainer` and call `fit`.
      # Q: Why don't we need a checkpoint callback?
      trainer = Trainer(max_epochs = self.config.train.optimizer.max_epochs) 
      trainer.fit(system, train_dataloader)
      # Call `predict` on `Trainer` and the test data loader.
      predicted_probs = trainer.predict(system, dataloaders = test_dataloader)
      # Q: What exactly is each method invocation doing in this line?
      predicted_probs = torch.cat(predicted_probs).squeeze(1).numpy()

      return predicted_probs

  @step
  def inspect(self):
    r"""Use confidence learning over examples to identify labels that 
    likely have issues with the `cleanlab` tool. 
    """
    prob = np.asarray(self.all_df.prob)
    prob = np.stack([1 - prob, prob]).T
  
    # rank label indices by issues
    ranked_label_issues = None
    
    # =============================
    # FILL ME OUT
    # 
    # Apply confidence learning to labels and out-of-sample
    # predicted probabilities. 
    # 
    # HINT: use cleanlab. See tutorial. 
    # 
    # Our solution is one function call.
    # 
    # Types
    # --
    # ranked_label_issues: List[int]
    # =============================
    # Q: How would we know that the y-values were stored in all_df.label as opposed to some other key?
    ranked_label_issues = find_label_issues(
      labels=np.asarray(self.all_df.label),
      pred_probs=prob,
      return_indices_ranked_by="self_confidence"
    )

    assert ranked_label_issues is not None, "`ranked_label_issues` not defined."

    # save this to class
    self.issues = ranked_label_issues
    print(f'{len(ranked_label_issues)} label issues found.')

    # Q: Shouldn't we be doing this in the .retrain_retest step, instead of here?
    # overwrite label for all the entries in all_df
    for index in self.issues:
      label = self.all_df.loc[index, 'label']
      # we FLIP the label!
      self.all_df.loc[index, 'label'] = 1 - label

    self.next(self.review)

  @step
  def review(self):
    r"""Format the data quality issues found such that they are ready to be 
    imported into LabelStudio. We expect the following format:

    [
      {
        "data": {
          "text": <review text>
        },
        "predictions": [
          {
            "value": {
              "choices": [
                  "Positive"
              ]
            },
            "from_name": "sentiment",
            "to_name": "text",
            "type": "choices"
          }
        ]
      }
    ]

    See https://labelstud.io/guide/predictions.html#Import-pre-annotations-for-text.and

    You do not need to complete anything in this function. However, look through the 
    code and make sure the operations and output make sense.
    """
    outputs = []
    for index in self.issues:
      row = self.all_df.iloc[index]
      output = {
        'data': {
          'text': str(row.review),
        },
        'predictions': [{
          'result': [
            {
              'value': {
                'choices': [
                  'Positive' if row.label == 1 else 'Negative'
                ]
              },
              'id': f'data-{index}',
              'from_name': 'sentiment',
              'to_name': 'text',
              'type': 'choices',
            },
          ],
        }],
      }
      outputs.append(output)

    # save to file
    preanno_path = join(self.config.review.save_dir, 'pre-annotations.json')
    to_json(outputs, preanno_path)

    self.next(self.retrain_retest)

  @step
  def retrain_retest(self):
    r"""Retrain without reviewing. Let's assume all the labels that 
    confidence learning suggested to flip are indeed erroneous."""

    # # ====================================
    # Q: I understand why we're revising the training data. But why are we revising the test data?
    revised_data_module = self._reannotate_unconfident_labels(
      data_module = ReviewDataModule(self.config),
      reannotated_dataframe = self.all_df
    )
    # start from scratch
    system = SentimentClassifierSystem(self.config)
    trainer = Trainer(max_epochs = self.config.train.optimizer.max_epochs)

    trainer.fit(system, revised_data_module)
    trainer.test(system, revised_data_module, ckpt_path = 'best')
    results = system.test_results

    self._print_and_log_results(results= results, log_filename= 'conflearn.json')

    self.next(self.end)

  def _reannotate_unconfident_labels(self, data_module, reannotated_dataframe):
    # We assume that we concatenated data in the strict order: training, then dev, then test
    # This assumption is met in the .crossval step
    train_size = len(data_module.train_dataset)
    dev_size = len(data_module.dev_dataset)

    # Q: Why do we need to write the dataframe's changes into the datamodule; why can't we just use the dataframe directly for our retraining?
    data_module.train_dataset.data = reannotated_dataframe.iloc[0:train_size]
    data_module.dev_dataset.data = reannotated_dataframe.iloc[train_size:(train_size + dev_size)]
    data_module.test_dataset.data = reannotated_dataframe.iloc[(train_size + dev_size):]

    return data_module

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python conflearn.py`. To list
  this flow, run `python conflearn.py show`. To execute
  this flow, run `python conflearn.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python conflearn.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python conflearn.py resume`
  
  You can specify a run id as well.
  """
  flow = TrainIdentifyReview()
