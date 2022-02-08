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
"""Utility functions and layers.
Code based on the implementation of "Multi-Modal Transformer":
https://github.com/gabeur/mmt
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ReduceDim(nn.Module):

  def __init__(self, input_dimension, output_dimension):
    super(ReduceDim, self).__init__()
    self.fc = nn.Linear(input_dimension, output_dimension)

  def forward(self, x):
    x = self.fc(x)
    x = F.normalize(x, dim=-1)
    return x


def cross_view_cls_inner_product(text_cls_embd,
                                 vid_cls_embd,
                                 merge_caption_similarities='avg',
                                 is_normalize=True):
  '''
  text_cls_embd: B x C x D
  vid_cls_embd: B x D
  '''
  b = vid_cls_embd.size(0)
  num_caps = text_cls_embd.size(1)
  text_cls_embd = text_cls_embd.view(b * num_caps, -1)

  if is_normalize:
    vid_cls_embd = F.normalize(vid_cls_embd, dim=-1)
    text_cls_embd = F.normalize(text_cls_embd, dim=-1)

  sims = th.matmul(text_cls_embd, vid_cls_embd.t())

  # aggregate similarities from different captions
  if merge_caption_similarities == 'avg':
    sims = sims.view(b, num_caps, b)
    sims = th.mean(sims, dim=1)
  elif merge_caption_similarities == 'indep':
    sims = sims.view(b * num_caps, b)
  else:
    msg = 'unrecognised merge mode: {}'
    raise ValueError(msg.format(merge_caption_similarities))
  return sims


def cross_view_vlad_inner_product(text_vlad_embds,
                                  vid_vlad_embds,
                                  merge_caption_similarities='avg',
                                  is_normalize=True):
  '''
  text_vlad_embds: B x Caps x K x D
  vid_vlad_embds: B x K x D
  '''
  text_vlad_embds = text_vlad_embds.permute(2,0,1,3)  # B x Caps x K x D -> K x B x Caps x D
  vid_vlad_embds = vid_vlad_embds.transpose(0,1)  # B x K x D -> K x B x D

  k = text_vlad_embds.shape[0]
  b = text_vlad_embds.shape[1]
  num_caps = text_vlad_embds.shape[2]
  
  if is_normalize:
    text_vlad_embds = F.normalize(text_vlad_embds, dim=-1)
    vid_vlad_embds = F.normalize(vid_vlad_embds, dim=-1)

  vlad_sims = th.einsum('ktcd,kvd->ktcv', text_vlad_embds, vid_vlad_embds)  # K x B(text) x Caps x B(vid)

  # aggregate similarities from different captions
  if merge_caption_similarities == 'avg':
    vlad_sims = vlad_sims.mean(2)  # K x B(text) x Caps x B(vid) -> K x B(text) x B(vid)
  elif merge_caption_similarities == 'indep':
    vlad_sims = vlad_sims.view(k, b * num_caps, b)
  else:
    msg = 'unrecognised merge mode: {}'
    raise ValueError(msg.format(merge_caption_similarities))
  
  return vlad_sims  # -> K x B x B


def sharded_cross_view_cls_inner_product(text_cls_code,
                                         vid_cls_code,
                                         merge_caption_similarities='avg'):
  '''
  text_cls_code: B x C x M x K
  vid_cls_code: B x M x K
  '''
  b = text_cls_code.shape[0]
  num_caps = text_cls_code.shape[1]
  m = text_cls_code.shape[2]
  k = text_cls_code.shape[3]

  # sims: M x B x C x B
  sims = th.einsum('tcmk,vmk->mtcv', text_cls_code, vid_cls_code)
  # aggregate similarities from different captions
  if merge_caption_similarities == 'avg':
    # sims: M x B x C x B -> M x B x B
    sims = th.mean(sims, dim=2)
  elif merge_caption_similarities == 'indep':
    # sims: M x B x C x B -> M x (B*C) x B
    sims = sims.view(m, b * num_caps, b)
  else:
    msg = 'unrecognised merge mode: {}'
    raise ValueError(msg.format(merge_caption_similarities))
  return sims


def sharded_cross_view_vlad_inner_product(text_vlad_codes,
                                          vid_vlad_codes,
                                          merge_caption_similarities='avg'):
  '''
  text_vlad_codes: B x C x Kvlad x M x K
  vid_vlad_codes: B x Kvlad x M x K
  '''
  b = text_vlad_codes.shape[0]
  num_caps = text_vlad_codes.shape[1]
  num_vlads = text_vlad_codes.shape[2]
  m = text_vlad_codes.shape[3]
  k = text_vlad_codes.shape[4]
  
  vlad_sims = th.einsum('tclmk,vlmk->lmtcv', # 'l': Kvlad
                        text_vlad_codes,
                        vid_vlad_codes)  # Kvlad x M x B(text) x Caps x B(vid)
  # aggregate similarities from different captions
  if merge_caption_similarities == 'avg':
    # vlad_sims: Kvlad x M x B(text) x Caps x B(vid) -> Kvlad x M x B x B
    vlad_sims = th.mean(vlad_sims, dim=-2)
  elif merge_caption_similarities == 'indep':
    # vlad_sims: Kvlad x M x B(text) x Caps x B(vid) -> Kvlad x M x (B*C) x B
    vlad_sims = vlad_sims.view(num_vlads, m, (b*num_caps), b)
  else:
    msg = 'unrecognised merge mode: {}'
    raise ValueError(msg.format(merge_caption_similarities))
  return vlad_sims
