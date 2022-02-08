# Copyright 2022 Jinpeng Wang
# Copyright 2020 Valentin Gabeur
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
"""Gating module.
Code based on the implementation of "Multi-Modal Transformer":
https://github.com/gabeur/mmt
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GatedEmbeddingUnit(nn.Module):
  """Gated embedding module.

  as described in
  "Learning a Text-Video Embedding from Incomplete and Heterogeneous Data"
  """

  def __init__(self, input_dimension, output_dimension, use_bn, normalize):
    super(GatedEmbeddingUnit, self).__init__()

    self.fc = nn.Linear(input_dimension, output_dimension)
    self.cg = ContextGating(output_dimension, add_batch_norm=use_bn)
    self.normalize = normalize

  def forward(self, x):
    x = self.fc(x)
    x = self.cg(x)
    if self.normalize:
      x = F.normalize(x, dim=-1)
    return x


class MimicCEGatedEmbeddingUnit(nn.Module):

  def __init__(self, input_dimension, output_dimension, use_bn):
    super().__init__()
    self.cg = ContextGating(input_dimension, add_batch_norm=use_bn)

  def forward(self, x):
    x = self.cg(x)
    x = F.normalize(x, dim=-1)
    return x


class GatedLinearUnit(nn.Module):

  def forward(self, x, mask):
    x = th.cat((x, mask), 1)
    return F.glu(x, 1)


class ContextGating(nn.Module):
  """Context gating class."""

  def __init__(self, dimension, add_batch_norm=True):
    super(ContextGating, self).__init__()
    self.fc = nn.Linear(dimension, dimension)
    self.add_batch_norm = add_batch_norm
    self.batch_norm = nn.BatchNorm1d(dimension)

  def forward(self, x):
    x1 = self.fc(x)
    if self.add_batch_norm:
      x1 = self.batch_norm(x1)
    x = th.cat((x, x1), 1)
    return F.glu(x, 1)


class GatedEmbeddingUnitReasoning(nn.Module):

  def __init__(self, output_dimension):
    super(GatedEmbeddingUnitReasoning, self).__init__()
    self.cg = ContextGatingReasoning(output_dimension)

  def forward(self, x, mask):
    x = self.cg(x, mask)
    x = F.normalize(x, dim=-1)
    return x


class ContextGatingReasoning(nn.Module):
  """Context gating reasoning class."""

  def __init__(self, dimension, add_batch_norm=True):
    super(ContextGatingReasoning, self).__init__()
    self.fc = nn.Linear(dimension, dimension)
    self.add_batch_norm = add_batch_norm
    self.batch_norm = nn.BatchNorm1d(dimension)
    self.batch_norm2 = nn.BatchNorm1d(dimension)

  def forward(self, x, x1):

    x2 = self.fc(x)

    if self.add_batch_norm:
      x1 = self.batch_norm(x1)
      x2 = self.batch_norm2(x2)

    t = x1 + x2

    x = th.cat((x, t), 1)
    return F.glu(x, 1)
