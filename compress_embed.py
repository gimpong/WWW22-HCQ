# Copyright 2022 Jinpeng Wang
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
"""Code for post-compressing text and video embeddings.
Code based on the implementation of "nanopq":
https://github.com/matsui528/nanopq
"""

import argparse
import model.metric as module_metric
import h5py as h5
import numpy as np
from scipy.cluster.vq import kmeans2, vq
from collections import defaultdict
from sklearn.decomposition import PCA

class LSH:
  def __init__(self, dim, num_bits):
    self.dim = dim
    self.num_bits = num_bits
  
  def fit(self, X):
    self.proj_mat = np.random.rand(self.dim, self.num_bits)

  def transform(self, X):
    return np.sign(X.dot(self.proj_mat))


class PQ(object):
  '''Codes from nanopq: https://github.com/matsui528/nanopq'''
  """Pure python implementation of Product Quantization (PQ) [Jegou11]_.

  For the indexing phase of database vectors,
  a `D`-dim input vector is divided into `M` `D`/`M`-dim sub-vectors.
  Each sub-vector is quantized into a small integer via `Ks` codewords.
  For the querying phase, given a new `D`-dim query vector, the distance beween the query
  and the database PQ-codes are efficiently approximated via Asymmetric Distance.

  All vectors must be np.ndarray with np.float32

  .. [Jegou11] H. Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011

  Args:
      M (int): The number of sub-space
      Ks (int): The number of codewords for each subspace
          (typically 256, so that each sub-vector is quantized
          into 256 bits = 1 byte = uint8)
      verbose (bool): Verbose flag

  Attributes:
      M (int): The number of sub-space
      Ks (int): The number of codewords for each subspace
      verbose (bool): Verbose flag
      code_dtype (object): dtype of PQ-code. Either np.uint{8, 16, 32}
      codewords (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32.
          codewords[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
      Ds (int): The dim of each sub-vector, i.e., Ds=D/M

  """

  def __init__(self, M, Ks=256, verbose=False):
    assert 0 < Ks <= 2 ** 32
    self.M, self.Ks, self.verbose = M, Ks, verbose
    self.code_dtype = (
      np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
    )
    self.codewords = None
    self.Ds = None

    if verbose:
      print("M: {}, Ks: {}, code_dtype: {}".format(M, Ks, self.code_dtype))

  def __eq__(self, other):
    if isinstance(other, PQ):
      return (self.M, self.Ks, self.verbose, self.code_dtype, self.Ds) == (
        other.M,
        other.Ks,
        other.verbose,
        other.code_dtype,
        other.Ds,
      ) and np.array_equal(self.codewords, other.codewords)
    else:
      return False

  def fit(self, vecs, iter=1, seed=2022):
    """Given training vectors, run k-means for each sub-space and create
    codewords for each sub-space.

    This function should be run once first of all.

    Args:
        vecs (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
        iter (int): The number of iteration for k-means
        seed (int): The seed for random process

    Returns:
        object: self

    """
    assert vecs.dtype == np.float32
    assert vecs.ndim == 2
    N, D = vecs.shape
    assert self.Ks < N, "the number of training vector should be more than Ks"
    assert D % self.M == 0, "input dimension must be dividable by M"
    self.Ds = int(D / self.M)

    np.random.seed(seed)
    if self.verbose:
      print("iter: {}, seed: {}".format(iter, seed))

    # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
    self.codewords = np.zeros((self.M, self.Ks, self.Ds), dtype=np.float32)
    for m in range(self.M):
      if self.verbose:
        print("Training the subspace: {} / {}".format(m, self.M))
      vecs_sub = vecs[:, m * self.Ds : (m + 1) * self.Ds]
      self.codewords[m], _ = kmeans2(vecs_sub, self.Ks, iter=iter, minit="points")

    return self

  def encode(self, vecs):
    """Encode input vectors into PQ-codes.

    Args:
        vecs (np.ndarray): Input vectors with shape=(N, D) and dtype=np.float32.

    Returns:
        np.ndarray: PQ codes with shape=(N, M) and dtype=self.code_dtype

    """
    assert vecs.dtype == np.float32
    assert vecs.ndim == 2
    N, D = vecs.shape
    assert D == self.Ds * self.M, "input dimension must be Ds * M"

    # codes[n][m] : code of n-th vec, m-th subspace
    codes = np.empty((N, self.M), dtype=self.code_dtype)
    for m in range(self.M):
      if self.verbose:
        print("Encoding the subspace: {} / {}".format(m, self.M))
      vecs_sub = vecs[:, m * self.Ds : (m + 1) * self.Ds]
      codes[:, m], _ = vq(vecs_sub, self.codewords[m])

    return codes

  def decode(self, codes):
    """Given PQ-codes, reconstruct original D-dimensional vectors
    approximately by fetching the codewords.

    Args:
        codes (np.ndarray): PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
            Each row is a PQ-code

    Returns:
        np.ndarray: Reconstructed vectors with shape=(N, D) and dtype=np.float32

    """
    assert codes.ndim == 2
    N, M = codes.shape
    assert M == self.M
    assert codes.dtype == self.code_dtype

    vecs = np.empty((N, self.Ds * self.M), dtype=np.float32)
    for m in range(self.M):
      vecs[:, m * self.Ds : (m + 1) * self.Ds] = self.codewords[m][codes[:, m], :]

    return vecs

  def transform(self, vecs):
    return self.decode(self.encode(vecs))


class OPQ(object):
  '''Codes from nanopq: https://github.com/matsui528/nanopq'''
  """Pure python implementation of Optimized Product Quantization (OPQ) [Ge14]_.
  OPQ is a simple extension of PQ.
  The best rotation matrix `R` is prepared using training vectors.
  Each input vector is rotated via `R`, then quantized into PQ-codes
  in the same manner as the original PQ.
  .. [Ge14] T. Ge et al., "Optimized Product Quantization", IEEE TPAMI 2014
  Args:
      M (int): The number of sub-spaces
      Ks (int): The number of codewords for each subspace (typically 256, so that each sub-vector is quantized
          into 256 bits = 1 byte = uint8)
      verbose (bool): Verbose flag
  Attributes:
      R (np.ndarray): Rotation matrix with the shape=(D, D) and dtype=np.float32
  """

  def __init__(self, M, Ks=256, verbose=False):
    self.pq = PQ(M, Ks, verbose)
    self.R = None

  def __eq__(self, other):
    if isinstance(other, OPQ):
      return self.pq == other.pq and np.array_equal(self.R, other.R)
    else:
      return False

  @property
  def M(self):
    """int: The number of sub-space"""
    return self.pq.M

  @property
  def Ks(self):
    """int: The number of codewords for each subspace"""
    return self.pq.Ks

  @property
  def verbose(self):
    """bool: Verbose flag"""
    return self.pq.verbose

  @verbose.setter
  def verbose(self, v):
    self.pq.verbose = v

  @property
  def code_dtype(self):
    """object: dtype of PQ-code. Either np.uint{8, 16, 32}"""
    return self.pq.code_dtype

  @property
  def codewords(self):
    """np.ndarray: shape=(M, Ks, Ds) with dtype=np.float32.
    codewords[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
    """
    return self.pq.codewords

  @property
  def Ds(self):
    """int: The dim of each sub-vector, i.e., Ds=D/M"""
    return self.pq.Ds

  def eigenvalue_allocation(self, vecs):
    """Given training vectors, this function learns a rotation matrix.
    The rotation matrix is computed so as to minimize the distortion bound of PQ,
    assuming a multivariate Gaussian distribution.
    This function is a translation from the original MATLAB implementation to that of python
    http://kaiminghe.com/cvpr13/index.html
    Args:
        vecs: (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
    Returns:
        R: (np.ndarray) rotation matrix of shape=(D, D) with dtype=np.float32.
    """
    _, D = vecs.shape
    cov = np.cov(vecs, rowvar=False)
    w, v = np.linalg.eig(cov)
    sort_ix = np.argsort(np.abs(w))[::-1]
    eig_vals = w[sort_ix]
    eig_vecs = v[:, sort_ix]

    assert D % self.M == 0, "input dimension must be dividable by M"
    Ds = D // self.M
    dim_tables = defaultdict(list)
    fvals = np.log(eig_vals + 1e-10)
    fvals = fvals - np.min(fvals) + 1
    sum_list = np.zeros(self.M)
    big_number = 1e10 + np.sum(fvals)

    cur_subidx = 0
    for d in range(D):
      dim_tables[cur_subidx].append(d)
      sum_list[cur_subidx] += fvals[d]
      if len(dim_tables[cur_subidx]) == Ds:
        sum_list[cur_subidx] = big_number
      cur_subidx = np.argmin(sum_list)

    dim_ordered = []
    for m in range(self.M):
      dim_ordered.extend(dim_tables[m])

    R = eig_vecs[:, dim_ordered]
    R = R.astype(dtype=np.float32)
    return R

  def fit(self, vecs, parametric_init=False, pq_iter=1, seed=2022):
    """Given training vectors, this function alternatively trains
    (a) codewords and (b) a rotation matrix.
    The procedure of training codewords is same as :func:`PQ.fit`.
    The rotation matrix is computed so as to minimize the quantization error
    given codewords (Orthogonal Procrustes problem)
    This function is a translation from the original MATLAB implementation to that of python
    http://kaiminghe.com/cvpr13/index.html
    If you find the error message is messy, please turn off the verbose flag, then
    you can see the reduction of error for each iteration clearly
    Args:
        vecs (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
        parametric_init (bool): Whether to initialize rotation using parametric assumption.
        pq_iter (int): The number of iteration for k-means
        seed (int): The seed for random process
    Returns:
        object: self
    """
    assert vecs.dtype == np.float32
    assert vecs.ndim == 2
    _, D = vecs.shape
    if parametric_init:
      self.R = self.eigenvalue_allocation(vecs)
    else:
      self.R = np.eye(D, dtype=np.float32)

    # (a) Train codewords
    pq_tmp = PQ(M=self.M, Ks=self.Ks, verbose=self.verbose)
    X = vecs @ self.R
    pq_tmp.fit(X, iter=pq_iter, seed=seed)
    # (b) Update a rotation matrix R
    X_ = pq_tmp.decode(pq_tmp.encode(X))
    U, s, V = np.linalg.svd(vecs.T @ X_)
    self.R = U @ V
    # (c) Re-train codewords
    self.pq = PQ(M=self.M, Ks=self.Ks, verbose=self.verbose)
    X = vecs @ self.R
    self.pq.fit(X, iter=pq_iter, seed=seed)

    return self

  def rotate(self, vecs):
    """Rotate input vector(s) by the rotation matrix.`
    Args:
        vecs (np.ndarray): Input vector(s) with dtype=np.float32.
            The shape can be a single vector (D, ) or several vectors (N, D)
    Returns:
        np.ndarray: Rotated vectors with the same shape and dtype to the input vecs.
    """
    assert vecs.dtype == np.float32
    assert vecs.ndim in [1, 2]

    if vecs.ndim == 2:
      return vecs @ self.R
    elif vecs.ndim == 1:
      return (vecs.reshape(1, -1) @ self.R).reshape(-1)

  def encode(self, vecs):
    """Rotate input vectors by :func:`OPQ.rotate`, then encode them via :func:`PQ.encode`.
    Args:
        vecs (np.ndarray): Input vectors with shape=(N, D) and dtype=np.float32.
    Returns:
        np.ndarray: PQ codes with shape=(N, M) and dtype=self.code_dtype
    """
    return self.pq.encode(self.rotate(vecs))

  def decode(self, codes):
    """Given PQ-codes, reconstruct original D-dimensional vectors via :func:`PQ.decode`,
    and applying an inverse-rotation.
    Args:
        codes (np.ndarray): PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
            Each row is a PQ-code
    Returns:
        np.ndarray: Reconstructed vectors with shape=(N, D) and dtype=np.float32
    """
    # Because R is a rotation matrix (R^t * R = I), R^-1 should be R^t
    return self.pq.decode(codes) @ self.R.T

  def transform(self, vecs):
    return self.decode(self.encode(vecs))


def calc_hammingSim(a, b):
  l = b.shape[-1]
  return (l + a.dot(b.T)) / 2


def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-6)
  
  
if __name__ == "main__":
  parser = argparse.ArgumentParser(description="PyTorch Template")
  parser.add_argument("--path",
                      default=None,
                      type=str,
                      help="path of embedding file")
  parser.add_argument("--type",
                      default='LSH',
                      type=str,
                      help="compression method")
  args = parser.parse_args()
  print(f"==============Embedding file path: {args.path}, compression type: {args.type}================")
  
  print("Loading and preparing files ...")
  data = h5.File(args.path)
  # training embeddings
  vid_cls_embd = data['train_vid_cls_embd'][()]
  vid_vlad_embds = data['train_vid_vlad_embds'][()]
  vid_vlad_embds = vid_vlad_embds.reshape(vid_vlad_embds.shape[0], -1)
  text_cls_embd = data['train_text_cls_embd'][()].squeeze()
  text_vlad_embds = data['train_text_vlad_embds'][()].squeeze()
  text_vlad_embds = text_vlad_embds.reshape(text_vlad_embds.shape[0], -1)
  vid_embd = np.concatenate([vid_cls_embd, vid_vlad_embds], axis=-1)
  text_embd = np.concatenate([text_cls_embd, text_vlad_embds], axis=-1)
  train_embd = np.concatenate([text_embd, vid_embd])
  
  # testing embeddings
  vid_cls_embd = data['test_vid_cls_embd'][()]
  vid_vlad_embds = data['test_vid_vlad_embds'][()]
  vid_vlad_embds = vid_vlad_embds.reshape(vid_vlad_embds.shape[0], -1)
  text_cls_embd = data['test_text_cls_embd'][()].squeeze()
  text_vlad_embds = data['test_text_vlad_embds'][()].squeeze()
  text_vlad_embds = text_vlad_embds.reshape(text_vlad_embds.shape[0], -1)
  test_vid_embd = np.concatenate([vid_cls_embd, vid_vlad_embds], axis=-1)
  test_text_embd = np.concatenate([text_cls_embd, text_vlad_embds], axis=-1)
  (n_samples, d) = test_vid_embd.shape
  data.close()

  if args.type == 'LSH':
    lsh = LSH(d, 2048)
    print("Begin to fit model ...")
    lsh.fit(train_embd)
    print("Finish fitting model and begin to transform testing data ...")
    proj_text_embd = lsh.transform(test_text_embd)
    proj_vid_embd = lsh.transform(test_vid_embd)
    print("Finish data transformation and begin to compute similarity")
    sims = calc_hammingSim(proj_text_embd, proj_vid_embd)
    print("Begin to compute metrics ... ")
    t2v_metr = module_metric.t2v_metrics(sims)
    v2t_metr = module_metric.v2t_metrics(sims) 
  elif args.type == 'PQ':
    pq = PQ(M=256)
    print("Begin to fit model ...")
    pq.fit(train_embd)
    print("Finish fitting model and begin to transform testing data ...")
    quant_text_code = pq.encode(test_text_embd)
    quant_vid_code = pq.encode(test_vid_embd)
    quant_text_embd = pq.decode(quant_text_code)
    quant_vid_embd = pq.decode(quant_vid_code)
    quant_text_embd = normalize(quant_text_embd.reshape((n_samples, d//512, 512))).reshape((n_samples, d))
    quant_vid_embd = normalize(quant_vid_embd.reshape((n_samples, d//512, 512))).reshape((n_samples, d))
    test_text_embd = normalize(test_text_embd.reshape((n_samples, d//512, 512))).reshape((n_samples, d))
    test_vid_embd = normalize(test_vid_embd.reshape((n_samples, d//512, 512))).reshape((n_samples, d))
    print("Finish data transformation and begin to compute similarity ...")
    quant_text_sims = quant_text_embd.dot(test_vid_embd.T)
    quant_vid_sims = test_text_embd.dot(quant_vid_embd.T)
    print("Begin to compute metrics ... ")
    t2v_metr = module_metric.t2v_metrics(quant_vid_sims)
    v2t_metr = module_metric.v2t_metrics(quant_text_sims) 
  elif args.type == 'OPQ':
    opq = OPQ(M=256)
    print("Begin to fit model ...")
    opq.fit(train_embd)
    print("Finish fitting model and begin to transform testing data ...")
    quant_text_code = opq.encode(test_text_embd)
    quant_vid_code = opq.encode(test_vid_embd)
    quant_text_embd = opq.decode(quant_text_code)
    quant_vid_embd = opq.decode(quant_vid_code)
    quant_text_embd = normalize(quant_text_embd.reshape((n_samples, d//512, 512))).reshape((n_samples, d))
    quant_vid_embd = normalize(quant_vid_embd.reshape((n_samples, d//512, 512))).reshape((n_samples, d))
    test_text_embd = normalize(test_text_embd.reshape((n_samples, d//512, 512))).reshape((n_samples, d))
    test_vid_embd = normalize(test_vid_embd.reshape((n_samples, d//512, 512))).reshape((n_samples, d))
    print("Finish data transformation and begin to compute similarity ...")
    quant_text_sims = quant_text_embd.dot(test_vid_embd.T)
    quant_vid_sims = test_text_embd.dot(quant_vid_embd.T)
    print("Begin to compute metrics ... ")
    t2v_metr = module_metric.t2v_metrics(quant_vid_sims)
    v2t_metr = module_metric.v2t_metrics(quant_text_sims) 
  else:
    raise NotImplementedError(f"the compression type {args.type} has not been implemented")
  
  # print results
  print("txt2vid: ")
  for metric_name in ["R1", "R5", "R10", "R50", "MedR", "MeanR", "geometric_mean_R1-R5-R10"]:
    print(' {:15s}: {}'.format(metric_name, t2v_metr[metric_name]))
  print("vid2txt: ")
  for metric_name in ["R1", "R5", "R10", "R50", "MedR", "MeanR", "geometric_mean_R1-R5-R10"]:
    print(' {:15s}: {}'.format(metric_name, v2t_metr[metric_name]))
  print("========================================================================================")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="PyTorch Template")
  parser.add_argument("--path",
                      default=None,
                      type=str,
                      help="path of embedding file")
  parser.add_argument("--type",
                      default='LSH',
                      type=str,
                      help="compression method")
  args = parser.parse_args()
  print(f"==============Embedding file path: {args.path}, compression type: {args.type}================")
  
  print("Loading and preparing files ...")
  data = h5.File(args.path)
  # training embeddings
  train_vid_cls_embd = data['train_vid_cls_embd'][()]
  train_vid_vlad_embds = data['train_vid_vlad_embds'][()]
  train_text_cls_embd = data['train_text_cls_embd'][()].squeeze()
  train_text_vlad_embds = data['train_text_vlad_embds'][()].squeeze()
  train_cls_embd = np.concatenate([train_text_cls_embd, train_vid_cls_embd])
  train_vlad_embds = np.concatenate([train_text_vlad_embds, train_vid_vlad_embds])
  
  # testing embeddings
  test_vid_cls_embd = data['test_vid_cls_embd'][()]
  test_vid_vlad_embds = data['test_vid_vlad_embds'][()]
  test_text_cls_embd = data['test_text_cls_embd'][()].squeeze()
  test_text_vlad_embds = data['test_text_vlad_embds'][()].squeeze()
  # test_cls_embd = np.concatenate([test_text_cls_embd, test_vid_cls_embd])
  # test_vlad_embds = np.concatenate([test_text_vlad_embds, test_vid_vlad_embds])
  (n_samples, num_vlad, d) = test_text_vlad_embds.shape
  data.close()
  
  assert args.type in ['LSH', 'PQ', 'OPQ']
  
  if args.type == 'LSH':
    model = {'cls': LSH(d, 256), 'vlad': [LSH(d, 256) for i in range(num_vlad)]}
  elif args.type == 'PQ':
    model = {'cls': PQ(M=32), 'vlad': [PQ(M=32) for i in range(num_vlad)]}
  else:
    model = {'cls': OPQ(M=32), 'vlad': [OPQ(M=32) for i in range(num_vlad)]}

  print("Begin to fit model ...")
  model['cls'].fit(train_cls_embd)
  for i in range(num_vlad):
    model['vlad'][i].fit(train_vlad_embds[:, i])
  
  print("Finish fitting model and begin to transform testing data ...")
  proj_text_cls_embd = model['cls'].transform(test_text_cls_embd)
  proj_vid_cls_embd = model['cls'].transform(test_vid_cls_embd)
  proj_vid_vlad_embds, proj_text_vlad_embds = [], []
  for i in range(num_vlad):
    proj_text_vlad_embds.append(model['vlad'][i].transform(test_text_vlad_embds[:, i]))
    proj_vid_vlad_embds.append(model['vlad'][i].transform(test_vid_vlad_embds[:, i]))
  
  print("Finish data transformation and begin to compute similarity")
  if args.type in ['LSH']:
    sims = calc_hammingSim(proj_text_cls_embd, proj_vid_cls_embd)
    for i in range(num_vlad):
      sims += calc_hammingSim(proj_text_vlad_embds[i], proj_vid_vlad_embds[i])
    sims /= (1 + num_vlad)
    print("Begin to compute metrics ... ")
    t2v_metr = module_metric.t2v_metrics(sims)
    v2t_metr = module_metric.v2t_metrics(sims)
  elif args.type in ['PQ', 'OPQ']:
    quant_vid_sims = normalize(test_text_cls_embd).dot(normalize(proj_vid_cls_embd).T)
    quant_text_sims = normalize(proj_text_cls_embd).dot(normalize(test_vid_cls_embd).T)
    for i in range(num_vlad):
      quant_vid_sims += normalize(test_text_vlad_embds[:, i]).dot(normalize(proj_vid_vlad_embds[i]).T)
      quant_text_sims += normalize(proj_text_vlad_embds[i]).dot(normalize(test_vid_vlad_embds[:, i]).T)
    quant_vid_sims /= (1 + num_vlad)
    quant_text_sims /= (1 + num_vlad)
    print("Begin to compute metrics ... ")
    t2v_metr = module_metric.t2v_metrics(quant_vid_sims)
    v2t_metr = module_metric.v2t_metrics(quant_text_sims) 
  
  # print results
  print("txt2vid: ")
  for metric_name in ["R1", "R5", "R10", "R50", "MedR", "MeanR", "geometric_mean_R1-R5-R10"]:
    print(' {:15s}: {}'.format(metric_name, t2v_metr[metric_name]))
  print("vid2txt: ")
  for metric_name in ["R1", "R5", "R10", "R50", "MedR", "MeanR", "geometric_mean_R1-R5-R10"]:
    print(' {:15s}: {}'.format(metric_name, v2t_metr[metric_name]))
  print("========================================================================================")


if __name__ == "main__":
  parser = argparse.ArgumentParser(description="PyTorch Template")
  parser.add_argument("--path",
                      default=None,
                      type=str,
                      help="path of embedding file")
  parser.add_argument("--cls_weight",
                      default=1/8,
                      type=float,
                      help="similarity weight of cls embedding")
  args = parser.parse_args()
  
  data = h5.File(args.path)
  vid_cls_embd = data['vid_cls_embd'][()]
  vid_vlad_embds = data['vid_vlad_embds'][()]
  text_cls_embd = data['text_cls_embd'][()].squeeze()
  text_vlad_embds = data['text_vlad_embds'][()].squeeze()
  cls_train_embd = np.concatenate([text_cls_embd, vid_cls_embd])
  vlad_train_embds = np.concatenate([text_vlad_embds, vid_vlad_embds])
  (n_samples, num_vlad, d) = vid_vlad_embds.shape
  data.close()
  
  lsh = {'cls': np.random.uniform(size=(d, 2048)),
         'vlad': [np.random.uniform(size=(d, 2048)) for i in range(num_vlad)]}
  proj_vid_cls_embd = np.sign(vid_cls_embd.dot(lsh['cls']))
  proj_text_cls_embd = np.sign(text_cls_embd.dot(lsh['cls']))
  proj_vid_vlad_embds, proj_text_vlad_embds = [], []
  for i in range(num_vlad):
    proj_vid_vlad_embds.append(np.sign(vid_vlad_embds[:, i].dot(lsh['vlad'][i])))
    proj_text_vlad_embds.append(np.sign(text_vlad_embds[:, i].dot(lsh['vlad'][i])))
  
  sims = np.zeros((n_samples, n_samples))
  sims += args.cls_weight * calc_hammingSim(proj_text_cls_embd, proj_vid_cls_embd)
  for i in range(num_vlad):
    sims += ((1 - args.cls_weight) * (1 / num_vlad) * calc_hammingSim(proj_text_vlad_embds[i], proj_vid_vlad_embds[i]))
  
  t2v_metr = module_metric.t2v_metrics(sims)
  v2t_metr = module_metric.v2t_metrics(sims)  
  
  # print results
  print("txt2vid: ")
  for metric_name in ["R1", "R5", "R10", "R50", "MedR", "MeanR", "geometric_mean_R1-R5-R10"]:
    print(' {:15s}: {}'.format(metric_name, t2v_metr[metric_name]))
  print("vid2txt: ")
  for metric_name in ["R1", "R5", "R10", "R50", "MedR", "MeanR", "geometric_mean_R1-R5-R10"]:
    print(' {:15s}: {}'.format(metric_name, v2t_metr[metric_name]))


if __name__ == "main__":
  parser = argparse.ArgumentParser(description="PyTorch Template")
  parser.add_argument("--path",
                      default=None,
                      type=str,
                      help="path of embedding file")
  parser.add_argument("--cls_weight",
                      default=1/8,
                      type=float,
                      help="similarity weight of cls embedding")
  args = parser.parse_args()
  
  data = h5.File(args.path)
  vid_cls_embd = data['vid_cls_embd'][()]
  vid_vlad_embds = data['vid_vlad_embds'][()]
  text_cls_embd = data['text_cls_embd'][()].squeeze()
  text_vlad_embds = data['text_vlad_embds'][()].squeeze()
  cls_train_embd = np.concatenate([text_cls_embd, vid_cls_embd])
  vlad_train_embds = np.concatenate([text_vlad_embds, vid_vlad_embds])
  (n_samples, num_vlad, d) = vid_vlad_embds.shape
  n_bits = 2048
  data.close()
  
  lsh = {'cls': PCA(512), # np.random.uniform(size=(d, 2048)),
         'vlad': [PCA(512) for i in range(num_vlad)]} # [np.random.uniform(size=(d, 2048)) for i in range(num_vlad)]}
  lsh['cls'].fit(cls_train_embd)
  for i in range(num_vlad):
    lsh['vlad'][i].fit(vlad_train_embds[:, i])
  proj_vid_cls_embd = np.sign(lsh['cls'].transform(vid_cls_embd))
  proj_text_cls_embd = np.sign(lsh['cls'].transform(text_cls_embd))
  proj_vid_vlad_embds, proj_text_vlad_embds = [], []
  for i in range(num_vlad):
    proj_vid_vlad_embds.append(np.sign(lsh['vlad'][i].transform(vid_vlad_embds[:, i])))
    proj_text_vlad_embds.append(np.sign(lsh['vlad'][i].transform(text_vlad_embds[:, i])))
  
  sims = np.zeros((n_samples, n_samples))
  sims += args.cls_weight * calc_hammingSim(proj_text_cls_embd, proj_vid_cls_embd)
  for i in range(num_vlad):
    sims += ((1 - args.cls_weight) * (1 / num_vlad) * calc_hammingSim(proj_text_vlad_embds[i], proj_vid_vlad_embds[i]))
  
  t2v_metr = module_metric.t2v_metrics(sims)
  v2t_metr = module_metric.v2t_metrics(sims)  
  
  # print results
  print("txt2vid: ")
  for metric_name in ["R1", "R5", "R10", "R50", "MedR", "MeanR", "geometric_mean_R1-R5-R10"]:
    print(' {:15s}: {}'.format(metric_name, t2v_metr[metric_name]))
  print("vid2txt: ")
  for metric_name in ["R1", "R5", "R10", "R50", "MedR", "MeanR", "geometric_mean_R1-R5-R10"]:
    print(' {:15s}: {}'.format(metric_name, v2t_metr[metric_name]))

