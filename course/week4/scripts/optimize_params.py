import time
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
from os import environ as env
from os import makedirs
from typing import List
from dotmap import DotMap
from dotenv import load_dotenv
from metaflow import FlowSpec, step, Parameter
from sentence_transformers import SentenceTransformer

from rag.paths import DATA_DIR
from rag.vector import retrieve_documents, get_my_collection_name

load_dotenv()

class OptimizeRagParams(FlowSpec):
  r"""MetaFlow to optimize a RAG hyperparameters by maximizing a retrieval 
  metric on top of an evaluation set.

  Arguments
  ---------
  questions_file (str, default: data/questions/questions.csv): path to generated questions CSV
  starpoint_api_key (str, default: env['STARPOINT_API_KEY']): Starpoint API key
  """
  questions_file = Parameter(
    'questions_file', 
    help = 'Path to generated questions', 
    default = join(DATA_DIR, 'questions/questions.csv'),
  )
  starpoint_api_key = Parameter(
    'starpoint_api_key', 
    help = 'Starpoint API key', 
    default = env['STARPOINT_API_KEY'],
  )

  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    self.next(self.get_search_space)

  @step 
  def get_search_space(self):
    r"""Define a set of RAG configurations to search over.
    """
    # ===========================
    embeddings = ["all-MiniLM-L6-v2", "thenlper/gte-small"]
    text_search_weights = [0, 0.5]
    hyde_embeddings = [False, True]

    hparams: List[DotMap] = []
    for e in embeddings:
      for tsw in text_search_weights:
        for he in hyde_embeddings:
          hparam = DotMap({
            "embedding": e,
            "text_search_weight": tsw,
            "hyde_embeddings": he,
          })

          hparams.append(hparam)

    assert len(hparams) > 0, "Remember to complete the code in `get_search_space`"
    assert len(hparams) == 8, "You should have 8 configurations" 
    self.hparams = hparams
    self.next(self.optimize, foreach='hparams')

  @step 
  def optimize(self):
    r"""Compute retrieval accuracy.
    :param hparam: Hyperparameter for this RAG retrieval system
    """
    print(f'Evaluating configuration: {self.input}')
    # Load the questions CSV containing generated questions and the 
    # doc id used to generate that question.
    questions = pd.read_csv(self.questions_file)

    # Use this to retrieve documents
    collection_name = get_my_collection_name(
      env['GITHUB_USERNAME'],
      embedding=self.input.embedding, 
      hyde=self.input.hyde_embeddings,
    )
    embedding_model = SentenceTransformer(self.input.embedding)

    hits = 0
    for i in tqdm(range(len(questions))):
      question = questions.question.iloc[i]
      groundtruth_doc_id = questions.doc_id.iloc[i]
      # ===========================
      retrieved_docs = retrieve_documents(
        api_key=self.starpoint_api_key,
        collection_name=collection_name,
        query=question,
        query_embedding= embedding_model.encode(question).tolist(),
        text_search_weight=self.input.text_search_weight,
        top_k=3,
      )
      hits += self._evaluate_hit(retrieved_docs, groundtruth_doc_id)

    hit_rate = hits / float(len(questions))
    self.hit_rate = hit_rate  # save to class
    self.hparam = self.input
    self.next(self.find_best)

  def _evaluate_hit(self, docs, ground_truth_doc_id):
    hit = 0

    if(any(doc['metadata']['doc_id'] == ground_truth_doc_id for doc in docs)):
      hit = 1

    return hit


  @step
  def find_best(self, inputs):
    r"""Given the outputs from the optimization, find the best hyperparameters
    by hit rate.
    """    
    save_dir = join(DATA_DIR, 'runs')
    makedirs(save_dir, exist_ok=True)

    results = []
    for input in inputs:
      result = {
        'hit_rate': input.hit_rate,
        **input.hparam,
      }
      results.append(result)

    results = pd.DataFrame.from_records(results)
    results.to_csv(join(save_dir, f'run-{int(time.time())}.csv'), index=False)
    
    self.next(self.end)

  @step
  def end(self):
    r"""End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python optimize_params.py`. To list
  this flow, run `python optimize_params.py show`. To execute
  this flow, run `python optimize_params.py --max-workers 1 run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python optimize_params.py --no-pylint --max-workers 1 run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python optimize_params.py resume`
  
  You can specify a run id as well.
  """
  flow = OptimizeRagParams()
