# Copyright 2022 Jinpeng Wang
# Copyright 2020 Valentin Gabeur
# Copyright 2020 Samuel Albanie, Yang Liu and Arsha Nagrani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for computing performance metrics.
Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts
Code based on the implementation of "Multi-Modal Transformer":
https://github.com/gabeur/mmt
"""

import ipdb
import numpy as np
import scipy.stats


def t2v_metrics(sims, query_masks=None):
  """Compute retrieval metrics from a similiarity matrix.

  Args:
    sims: N x M matrix of similarities between embeddings, where x_{i,j} =
      <text_embd[i], vid_embed[j]>
    query_masks: mask any missing queries from the dataset (two videos in MSRVTT
      only have 19, rather than 20 captions)

  Returns:
    (dict[str:float]): retrieval metrics
  """
  assert sims.ndim == 2, "expected a matrix, but currently sims.size = {}".format(sims.shape)
  nq, nv = sims.shape
  dists = -sims
  sorted_dists = np.sort(dists, axis=1)

  # The indices are computed such that they slice out the ground truth distances
  # from the psuedo-rectangular dist matrix
  qu = nq // nv  # Nb of queries per video
  gt_idx = [[np.ravel_multi_index([ii, jj], (nq, nv)) for ii in range(jj * qu, (jj + 1) * qu)] for jj in range(nv)]
  gt_idx = np.array(gt_idx)
  gt_dists = dists.reshape(-1)[gt_idx.reshape(-1)]
  gt_dists = gt_dists[:, np.newaxis]
  rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

  # --------------------------------
  # NOTE: Breaking ties
  # --------------------------------
  # We sometimes need to break ties (in general, these should occur extremely
  # rarely, but there are pathological cases when they can distort the scores,
  # such as when the similarity matrix is all zeros). Previous implementations
  # (e.g. the t2i evaluation function used here:
  # https://github.com/niluthpol/multimodal_vtt/blob/master/evaluation.py
  # and here:
  # https://github.com/linxd5/VSE_Pytorch/blob/master/evaluation.py#L87)
  # generally break ties "optimistically". However, if the similarity matrix is
  # constant this can evaluate to a perfect ranking. A principled option is to
  # average over all possible partial orderings implied by the ties. See this
  # paper for a discussion:
  #    McSherry, Frank, and Marc Najork,
  #    "Computing information retrieval performance measures efficiently in the
  #    presence of tied scores."
  #    European conference on information retrieval. Springer, Berlin,
  #    Heidelberg, 2008.
  # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.145.8892&rep=rep1&type=pdf

  # break_ties = "optimistically"
  break_ties = "averaging"

  if rows.size > nq:
    assert np.unique(rows).size == nq, "issue in metric evaluation"
    if break_ties == "optimistically":
      _, idx = np.unique(rows, return_index=True)
      cols = cols[idx]
    elif break_ties == "averaging":
      # fast implementation, based on this code:
      # https://stackoverflow.com/a/49239335
      locs = np.argwhere((sorted_dists - gt_dists) == 0)

      # Find the split indices
      steps = np.diff(locs[:, 0])
      splits = np.nonzero(steps)[0] + 1
      splits = np.insert(splits, 0, 0)

      # Compute the result columns
      summed_cols = np.add.reduceat(locs[:, 1], splits)
      counts = np.diff(np.append(splits, locs.shape[0]))
      avg_cols = summed_cols / counts
      cols = avg_cols

  msg = "expected ranks to match queries ({} vs {}) "
  if cols.size != nq:
    ipdb.set_trace()
  assert cols.size == nq, msg

  if query_masks is not None:
    # remove invalid queries
    assert query_masks.size == nq, "invalid query mask shape: {}, it should be {}".format(query_masks.size, nq)
    cols = cols[query_masks.reshape(-1).astype(np.bool)]
    assert cols.size == query_masks.sum(), "masking was not applied correctly"
    # update number of queries to account for those that were missing
    nq = query_masks.sum()
    
  return cols2metrics(cols, nq)


def v2t_metrics(sims, query_masks=None):
  """Compute retrieval metrics from a similiarity matrix.

  Args:
    sims: N x M matrix of similarities between embeddings, where
      x_{i,j} = <text_embd[i], vid_embed[j]>
    query_masks: mask any missing captions from the dataset

  Returns:
    (dict[str:float]): retrieval metrics

  NOTES: We find the closest "GT caption" in the style of VSE, which
  corresponds
  to finding the rank of the closest relevant caption in embedding space:
  github.com/ryankiros/visual-semantic-embedding/blob/master/evaluation.py#L52-L56
  """
  # switch axes of text and video
  sims = sims.T

  assert sims.ndim == 2, "expected a matrix"
  num_queries, num_caps = sims.shape
  dists = -sims
  caps_per_video = num_caps // num_queries
  break_ties = "averaging"

  missing_val = 1E8
  query_ranks = []
  for ii in range(num_queries):
    row_dists = dists[ii, :]
    if query_masks is not None:
      # Set missing queries to have a distance of infinity.  A missing query
      # refers to a query position `n` for a video that had less than `n`
      # captions (for example, a few MSRVTT videos only have 19 queries)
      row_dists[np.logical_not(query_masks.reshape(-1))] = missing_val

    # NOTE: Using distance subtraction to perform the ranking is easier to make
    # deterministic than using argsort, which suffers from the issue of defining
    # "stability" for equal distances.  Example of distance subtraction code:
    # github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/train.py
    sorted_dists = np.sort(row_dists)

    min_rank = np.inf
    for jj in range(ii * caps_per_video, (ii + 1) * caps_per_video):
      if row_dists[jj] == missing_val:
        # skip rankings of missing captions
        continue
      ranks = np.where((sorted_dists - row_dists[jj]) == 0)[0]
      if break_ties == "optimistically":
        rank = ranks[0]
      elif break_ties == "averaging":
        # NOTE: If there is more than one caption per video, its possible for
        # the method to do "worse than chance" in the degenerate case when all
        # similarities are tied.
        rank = ranks.mean()
      if rank < min_rank:
        min_rank = rank
    query_ranks.append(min_rank)
  query_ranks = np.array(query_ranks)

  return cols2metrics(query_ranks, num_queries)


def cols2metrics(cols, num_queries):
  """Compute the metrics."""
  metrics = {}
  metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
  metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
  metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
  metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries
  metrics["MedR"] = np.median(cols) + 1
  metrics["MeanR"] = np.mean(cols) + 1
  stats = [metrics[x] for x in ("R1", "R5", "R10")]
  metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
  metrics["cols"] = [int(i) for i in list(cols)]
  return metrics
