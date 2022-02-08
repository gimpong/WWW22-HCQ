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
# pylint: disable=g-explicit-length-test
"""Models base class.
Code based on the implementation of "Multi-Modal Transformer":
https://github.com/gabeur/mmt
"""
import abc

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
  """Base class for all models."""

  @abc.abstractmethod
  def forward(self, *inputs):
    """Forward pass logic."""
    raise NotImplementedError

  def __str__(self):
    """Model prints with number of trainable parameters."""
    model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return super().__str__() + f"\nTrainable parameters: {params}"
