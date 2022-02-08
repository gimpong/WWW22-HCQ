# Copyright 2022 Jinpeng Wang
# Copyright 2020 Valentin Gabeur
# Copyright 2020 Samuel Albanie, Yang Liu and Arsha Nagrani
# Copyright 2018 Antoine Miech All Rights Reserved.
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
"""Training losses.
Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts
Code based on the implementation of "Mixture of Embedding Experts":
https://github.com/antoine77340/Mixture-of-Embedding-Experts
Code based on the implementation of "Multi-Modal Transformer":
https://github.com/gabeur/mmt
"""
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MaxMarginRankingLoss(nn.Module):
  """Implementation of the Max-margin ranking loss."""

  def __init__(self, margin=1, fix_norm=True):
    super().__init__()
    self.fix_norm = fix_norm
    self.loss = th.nn.MarginRankingLoss(margin)
    self.margin = margin

  def forward(self, x):
    n = x.size()[0]

    x1 = th.diag(x)
    x1 = x1.unsqueeze(1)
    x1 = x1.expand(n, n)
    x1 = x1.contiguous().view(-1, 1)
    x1 = th.cat((x1, x1), 0)

    x2 = x.view(-1, 1)
    x3 = x.transpose(0, 1).contiguous().view(-1, 1)

    x2 = th.cat((x2, x3), 0)
    max_margin = F.relu(self.margin - (x1 - x2))

    if self.fix_norm:
      # remove the elements from the diagonal
      keep = th.ones(x.shape) - th.eye(x.shape[0])
      keep1 = keep.view(-1, 1)
      keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
      keep_idx = th.nonzero(th.cat((keep1, keep2), 0).flatten()).flatten()
      if x1.is_cuda:
        keep_idx = keep_idx.cuda()
      x1_ = th.index_select(x1, dim=0, index=keep_idx)
      x2_ = th.index_select(x2, dim=0, index=keep_idx)
      max_margin = F.relu(self.margin - (x1_ - x2_))

    return max_margin.mean()


class InfoNCELoss(nn.Module):
  def __init__(self, 
               t2v_temperature, 
               v2t_temperature):
    super().__init__()
    self.loss = th.nn.CrossEntropyLoss(reduction='mean')
    self.t2v_temperature = t2v_temperature
    self.v2t_temperature = v2t_temperature

  def forward(self, t2v_sims, v2t_sims):
    b = t2v_sims.shape[-2]
    target = th.arange(b)
    target = target.to(t2v_sims.device)

    if t2v_sims.ndim == 2:
      pass
    elif t2v_sims.ndim == 3:  # -> M x B x B
      m = t2v_sims.shape[0]
      target = target.repeat(m)
      t2v_sims = t2v_sims.view(m * b, b)
      v2t_sims = v2t_sims.reshape(m * b, b)  # == .contiguous().view(m*b, b)

    return self.loss(t2v_sims / self.t2v_temperature, target) + \
      self.loss(v2t_sims / self.v2t_temperature, target)


class DebiasedInfoNCELoss(nn.Module):
  def __init__(self, 
               t2v_temperature, 
               v2t_temperature,
               t2v_debias_prior, 
               v2t_debias_prior):
    super().__init__()
    self.t2v_temperature = t2v_temperature
    self.v2t_temperature = v2t_temperature
    self.t2v_debias_prior = t2v_debias_prior
    self.v2t_debias_prior = v2t_debias_prior

  def _debiased_simclr_loss(self, sims, temperature, pos_prior, mode='debias'):
    if sims.ndim == 2:
      cur_batch_size = sims.shape[0]  # -> B x B
      labels = th.eye(cur_batch_size).to(sims.device)  # -> B x B
      pos_logits = sims[labels.bool()]  # -> B
      neg_logits = sims[~labels.bool()].view(cur_batch_size, -1)  # -> B x (B-1)
    elif sims.ndim == 3:
      num_clusters = sims.shape[0]  # -> Kvlad x B x B
      cur_batch_size = sims.shape[1]
      labels = th.eye(cur_batch_size).to(sims.device)  # -> B x B
      pos_logits = sims[:,labels.bool()]  # -> Kvlad x B
      neg_logits = sims[:,~labels.bool()].view(num_clusters, cur_batch_size, -1) # Kvlad x B x (B-1)
    elif sims.ndim == 4:
      num_clusters = sims.shape[0]  # -> Kvlad x M x B x B
      num_subcodebooks = sims.shape[1]
      cur_batch_size = sims.shape[2]
      labels = th.eye(cur_batch_size).to(sims.device)  # -> B x B
      pos_logits = sims[:,:,labels.bool()]  # -> Kvlad x M x B
      neg_logits = sims[:,:,~labels.bool()].view(num_clusters, num_subcodebooks, cur_batch_size, -1) # Kvlad x M x B x (B-1)
    else:
      raise RuntimeError(f"Invalid dimensionality of similarity matrix: {sims.ndim}!")

    pos_probs = (pos_logits / temperature).exp()
    neg_probs = (neg_logits / temperature).exp()

    if mode == 'debias':
      N = cur_batch_size - 1
      Ng = th.clamp((-pos_prior * N * pos_probs + neg_probs.sum(dim=-1)) / (1 - pos_prior),
                    min=math.exp(N * (-1 / temperature)))  # -> Kvlad x B
    else:  # 'simple'
      Ng = neg_probs.sum(dim=-1)

    loss = (- th.log(pos_probs / (pos_probs + Ng))).mean(-1)  # -> Kvlad
    return loss

  def forward(self, t2v_sims, v2t_sims):
    t2v_loss = self._debiased_simclr_loss(sims=t2v_sims, 
                                          temperature=self.t2v_temperature,
                                          pos_prior=self.t2v_debias_prior)
    v2t_loss = self._debiased_simclr_loss(sims=v2t_sims, 
                                          temperature=self.v2t_temperature,
                                          pos_prior=self.v2t_debias_prior)
    return t2v_loss + v2t_loss


class HybridContrastiveLoss(nn.Module):
  def __init__(self, 
               cls_loss_hparams, 
               vlad_loss_hparams, 
               cls_loss_weight,
               vlad_loss_weight,
               code_loss_weight, 
               total_epochs, 
               pq_reg_loss_weight):
    super().__init__()
    self.quant_cls_loss = InfoNCELoss(**cls_loss_hparams)
    self.quant_vlad_loss = DebiasedInfoNCELoss(**vlad_loss_hparams)
    self.code_cls_loss = InfoNCELoss(**cls_loss_hparams)
    self.code_vlad_loss = DebiasedInfoNCELoss(**vlad_loss_hparams)
    self.feat_cls_loss = InfoNCELoss(**cls_loss_hparams)
    self.feat_vlad_loss = DebiasedInfoNCELoss(**vlad_loss_hparams)
    self.cls_loss_weight = cls_loss_weight
    self.vlad_loss_weight = vlad_loss_weight
    self.code_loss_weight = code_loss_weight
    self.total_epochs = total_epochs
    self._smoothing_weight_list = [math.cos(e * math.pi / 2 / total_epochs) for e in range(total_epochs + 1)]
    self._pq_reg_loss_weight = pq_reg_loss_weight
  
  @property
  def pq_reg_loss_weight(self):
    return self._pq_reg_loss_weight

  def smoothing_schedule(self, epoch):
    return self._smoothing_weight_list[epoch]

  def forward(self, 
              quant_cls_t2v_sims, 
              quant_cls_v2t_sims,
              quant_vlad_t2v_sims, 
              quant_vlad_v2t_sims, 
              feat_cls_t2v_sims, 
              feat_cls_v2t_sims, 
              feat_vlad_t2v_sims, 
              feat_vlad_v2t_sims,
              code_cls_t2v_sims,
              code_cls_v2t_sims,
              code_vlad_t2v_sims,
              code_vlad_v2t_sims,
              epoch):

    quant_cls_term = self.quant_cls_loss(t2v_sims=quant_cls_t2v_sims, 
                                         v2t_sims=quant_cls_v2t_sims)
    quant_vlad_term = self.quant_vlad_loss(t2v_sims=quant_vlad_t2v_sims, 
                                           v2t_sims=quant_vlad_v2t_sims).mean(0)  # -> B
    code_cls_term = self.code_cls_loss(t2v_sims=code_cls_t2v_sims,
                                       v2t_sims=code_cls_v2t_sims)  # -> B
    code_vlad_term = self.code_vlad_loss(t2v_sims=code_vlad_t2v_sims,
                                         v2t_sims=code_vlad_v2t_sims).mean(0).mean(0)  # -> B
    feat_cls_term = self.feat_cls_loss(t2v_sims=feat_cls_t2v_sims,
                                       v2t_sims=feat_cls_v2t_sims)
    feat_vlad_term = self.feat_vlad_loss(t2v_sims=feat_vlad_t2v_sims,
                                         v2t_sims=feat_vlad_v2t_sims).mean(0)  # -> B

    smoothing_weight = self.smoothing_schedule(epoch)

    cls_term = (quant_cls_term + code_cls_term * self.code_loss_weight) * (1 - smoothing_weight) + \
      feat_cls_term * smoothing_weight
    vlad_term = (quant_vlad_term + code_vlad_term * self.code_loss_weight) * (1 - smoothing_weight) + \
      feat_vlad_term * smoothing_weight

    return self.cls_loss_weight * cls_term + self.vlad_loss_weight * vlad_term


class FeatureBasedHybridContrastiveLoss(nn.Module):
  def __init__(self, 
               cls_loss_hparams, 
               vlad_loss_hparams, 
               cls_loss_weight,
               vlad_loss_weight):
    super().__init__()
    self.feat_cls_loss = InfoNCELoss(**cls_loss_hparams)
    self.feat_vlad_loss = DebiasedInfoNCELoss(**vlad_loss_hparams)
    self.cls_loss_weight = cls_loss_weight
    self.vlad_loss_weight = vlad_loss_weight
    
  def forward(self, 
              feat_cls_t2v_sims, 
              feat_cls_v2t_sims, 
              feat_vlad_t2v_sims, 
              feat_vlad_v2t_sims):

    feat_cls_term = self.feat_cls_loss(t2v_sims=feat_cls_t2v_sims,
                                       v2t_sims=feat_cls_v2t_sims)
    feat_vlad_term = self.feat_vlad_loss(t2v_sims=feat_vlad_t2v_sims,
                                         v2t_sims=feat_vlad_v2t_sims).mean(0)  # -> B

    return self.cls_loss_weight * feat_cls_term + self.vlad_loss_weight * feat_vlad_term


class DCMHLoss(nn.Module):
  def __init__(self, 
               scaling_factor, 
               quant_loss_weight, 
               balance_loss_weight):
    super().__init__()
    self.scaling_factor = scaling_factor
    self.quant_loss_weight = quant_loss_weight
    self.balance_loss_weight = balance_loss_weight

  def forward(self, text_cls_embd, vid_cls_embd):
    b = text_cls_embd.size(0)
    num_caps = text_cls_embd.size(1)
    n_bits = text_cls_embd.size(2)
    text_cls_embd = text_cls_embd.view(b * num_caps, -1)

    sims = th.matmul(F.normalize(text_cls_embd, dim=-1),
                     F.normalize(vid_cls_embd, dim=-1).t())
    sims = sims.view(b, num_caps, b)
    sims = th.mean(sims, dim=1)
    
    semantic_loss = (1 + (self.scaling_factor * sims).exp()).log().sum() - self.scaling_factor * sims.diag().sum()
    semantic_loss /= (sims.size(0) * sims.size(1))
    square_fn = lambda x: x * x
    binary_codes = th.sign(text_cls_embd + vid_cls_embd)
    quant_loss = (square_fn(binary_codes - text_cls_embd) + square_fn(binary_codes - vid_cls_embd)).mean()
    bit_balance_loss = (square_fn(text_cls_embd.mean(0)) + square_fn(vid_cls_embd.mean(0))).mean()
    total_loss = semantic_loss + self.quant_loss_weight * quant_loss + self.balance_loss_weight * bit_balance_loss
    return total_loss, {'total_loss': total_loss.item(), 
                        'semantic_loss': semantic_loss.item(), 
                        'quant_loss': quant_loss.item(), 
                        'bit_balance_loss': bit_balance_loss.item()}


class QuantizedRankingLoss(nn.Module):
  def __init__(self, 
               margin, 
               total_epochs, 
               pq_reg_loss_weight):
    super().__init__()
    self.margin = margin
    self.total_epochs = total_epochs
    self._smoothing_weight_list = [math.cos(e * math.pi / 2 / total_epochs) for e in range(total_epochs + 1)]
    self._pq_reg_loss_weight = pq_reg_loss_weight
  
  @property
  def pq_reg_loss_weight(self):
    return self._pq_reg_loss_weight

  def smoothing_schedule(self, epoch):
    return self._smoothing_weight_list[epoch]

  def forward(self, 
              quant_cls_t2v_sims, 
              quant_cls_v2t_sims,
              epoch):

    return self._rank_loss(quant_cls_t2v_sims) + self._rank_loss(quant_cls_v2t_sims)

  def _rank_loss(self, sims):
    b = sims.size(0)
    pos_sims = sims.diag().unsqueeze(-1)
    max_margin = F.relu(self.margin - (pos_sims - sims))
    neg_mask = ~th.eye(b, dtype=th.bool).to(sims.device)
    return max_margin[neg_mask].mean()

