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
"""Trainable quantization implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def tensor_square(inp):
    return inp * inp

class PQLayer(nn.Module):
    def __init__(self, feat_dim, M, K, 
                 mask_topK=None, alpha=1, 
                 update_codebooks=False, 
                 dual_codebooks=False,
                 add_BN=True,
                 pq_detach_opt='features'):
        '''
        mode='SX': dual_codebooks=True
        mode='VQ': dual_codebooks=False
        '''
        super(PQLayer, self).__init__()
        self.feat_dim, self.M, self.K, self.D = feat_dim, M, K, feat_dim//M
        self.mask_topK = mask_topK
        self.alpha = alpha
        self.dual_codebooks = dual_codebooks
        self._C_k = nn.Parameter(torch.empty((self.M, self.K, self.D)), requires_grad=update_codebooks)
        nn.init.xavier_uniform_(self._C_k.data)
        if dual_codebooks:
            self._C_v = nn.Parameter(torch.empty((self.M, self.K, self.D)), requires_grad=update_codebooks)
            nn.init.xavier_uniform_(self._C_v.data)
        else:
            self._C_v = self._C_k
        self._codebook_normalization()
        self.BatchNorm = nn.BatchNorm1d(self.M, affine=False) if add_BN else None
        
        assert pq_detach_opt in ['features', 'codewords', 'none']
        self.pq_detach_opt = pq_detach_opt

    @torch.no_grad()
    def _codebook_normalization(self):
        # normalize the codewords
        codewords_k = self._C_k.data.clone()
        codewords_k = F.normalize(codewords_k, dim=-1)
        self._C_k.copy_(codewords_k)
        if self.dual_codebooks:
            codewords_v = self._C_v.data.clone()
            codewords_v = F.normalize(codewords_v, dim=-1)
            self._C_v.copy_(codewords_v)

    @torch.no_grad()
    def _update_codebooks(self, text_embd, vid_embd, iter_nums=10, threshold=1e-5):
        '''
        text_embd: B x Caps x D, vid_embd: B x D
        '''
        assert not self.dual_codebooks, 'only VQ mode supports codebooks update'
        B = text_embd.shape[0]
        C = text_embd.shape[1]
        D = text_embd.shape[2]
        embds = torch.cat([text_embd.view(B * C, D), vid_embd], dim=0).view(B * C + B, 
                                                                            self.M, 
                                                                            self.D) # (BxCap+B) x M x d
        embds = F.normalize(embds, dim=-1)
        codewords_k = self._C_k.data.clone() # M x K x d
        codewords_k = F.normalize(codewords_k, dim=-1)
        for idx in range(self.M):
            embd = embds[:, idx] # (BxCap+B) x d
            sub_codewords = codewords_k[idx] # K x d
            tmp_sub_codewords = torch.zeros_like(sub_codewords) # K x d
            for it in range(iter_nums):
                sims = torch.matmul(embd, sub_codewords.t()) # (BxCap+B) x K
                cluster_labels = sims.argmax(dim=-1)
                tmp_sub_codewords.zero_()
                tmp_sub_codewords.scatter_add_(0, 
                                               cluster_labels.unsqueeze(-1).repeat(1, self.D), 
                                               embd)
                N_labels = torch.bincount(cluster_labels, minlength=self.K).type_as(tmp_sub_codewords).view(self.K, 1)
                tmp_sub_codewords /= (N_labels + 1e-6)
                tmp_sub_codewords = F.normalize(tmp_sub_codewords, dim=-1)
                if F.mse_loss(sub_codewords, tmp_sub_codewords) <= threshold:
                    break
                sub_codewords = tmp_sub_codewords
            codewords_k[idx] = sub_codewords
        self._C_k.copy_(codewords_k)
        del embd
        del sims
        del sub_codewords
        del tmp_sub_codewords
        del cluster_labels
        del N_labels
        del codewords_k

    def reconstruct(self, codes, hard_quant=False):
        # self._codebook_normalization()
        if hard_quant:
            # codesT:[Mxb]
            codesT = codes.T
            # x_hat:[bxMxD]: self._C_k:[MxKxD].lookup(codesT:[Mxb])
            x_hat_ = []
            for i in range(self.M):
                # x_hat_[i]:[bxD]: _C[i]:[KxD].lookup(codesT[i]:[b])
                x_hat_.append(self._C_k[i][codesT[i]])
            # x_hat_:[MxbxD]=>[bxMxD]
            x_hat_ = torch.transpose(torch.stack(x_hat_), 0, 1)
            # print("x_hat_:\n", x_hat_, "\nshape[bxMxD]=", x_hat_.shape, end='\n\n')
            # x_hat:[bxMxD]=>[bxfeat_dim]
            x_hat = x_hat_.reshape(x_hat_.shape[0], -1)
        else: # soft assignment
            # x_hat_:[bxMxD]
            # _C:[MxKxD], codes:[bxMxK] => x_hat_:[bxMxD]
            x_hat_ = torch.einsum('mkd,bmk->bmd', self._C_k, codes)
            # print("x_hat_:\n", x_hat_, "\nshape[bxMxD]=", x_hat_.shape, end='\n\n')
            # x_hat:[bxMxD]=>[bxfeat_dim]
            x_hat = x_hat_.view(x_hat_.shape[0], -1)

        return x_hat

    def forward(self, x):
        if x.ndim == 3: # bxexd
            collection_list = []
            x_T = torch.transpose(x, 0, 1) # exbxd
            # cap == 0
            out = self.forward(x_T[0])
            for i in range(len(out)):
                collection_list.append([out[i]])
            # cap > 0
            for cap in range(1, x_T.size(0)): # caption
                out = self.forward(x_T[cap])
                for i in range(len(out)):
                    collection_list[i].append(out[i])
            for i in range(len(collection_list)):
                # [(m)xbx...] -> [bxmx...]
                collection_list[i] = torch.stack(collection_list[i], 1)

            # x_hat: [bxcapsxfeat_dim], hard_codes: [bxcapsxM], soft_codes: [bxcapsxMxK]
            # x_normalized: [bxcapsxfeat_dim], reg_loss: [bxcaps]
            return collection_list

        elif x.ndim == 2:
            # x:[bxd]=>[bxMxD]
            x_ = x.view(x.shape[0], self.M, self.D)
            x_normalized_ = F.normalize(x_, dim=-1)
            x_normalized = x_normalized_.view(x.shape[0], self.feat_dim)
            # x_normalized_:[bxMxD], _C:[MxKxD] => sims:[bxMxK]
            sims = torch.einsum('bmd,mkd->bmk', x_normalized_, self._C_k)

            if self.BatchNorm:
                sims = self.BatchNorm(sims)

            # hard_codes:[bxM]
            hard_codes = sims.argmax(dim=-1)
            # hard_codesT:[Mxb]
            hard_codesT = hard_codes.T
            # x_hat:[bxMxD]: _C_v:[MxKxD].lookup(hard_codesT:[Mxb])
            x_hat_hard_ = []
            for i in range(self.M):
                # x_hat_hard_[i]:[bxD]: _C_v[i]:[KxD].lookup(hard_codesT[i]:[b])
                x_hat_hard_.append(self._C_v[i][hard_codesT[i]])
            # x_hat_hard_:[MxbxD]=>[bxMxD]
            x_hat_hard_ = torch.transpose(torch.stack(x_hat_hard_), -3, -2)
            # x_hat_hard:[bxMxD]=>[bxfeat_dim]
            x_hat_hard = x_hat_hard_.reshape(x_hat_hard_.shape[0], -1)

            # soft assignment 
            # soft_codes:[bxMxK]
            if self.mask_topK is not None:
                mask = torch.ones_like(sims, dtype=torch.bool)
                _, topK_ids = torch.topk(sims, k=self.mask_topK, dim=-1)
                mask.scatter_(-1, topK_ids, False)
                masked_sims = sims.masked_fill(mask, -np.inf)
                soft_codes = F.softmax(masked_sims * self.alpha, dim=-1)
            else:
                soft_codes = F.softmax(sims * self.alpha, dim=-1)

            # x_hat_soft_:[bxMxD]
            # _C:[MxKxD], soft_codes:[bxMxK] => x_hat_:[bxMxD]
            x_hat_soft_ = torch.einsum('mkd,bmk->bmd', self._C_v, soft_codes)
            # x_hat_soft:[bxMxD]=>[bxfeat_dim]
            x_hat_soft = x_hat_soft_.view(x_hat_soft_.shape[0], -1)

            if self.dual_codebooks:
                x_hat = x_hat_soft - (x_hat_soft - x_hat_hard).detach()
            else:
                x_hat = x_normalized - (x_normalized - x_hat_hard).detach()

            if self.dual_codebooks:
                # entropy regularization
                # softcodes: [bxMxK], reg_loss: [b]
                reg_loss = - (soft_codes * (soft_codes + 1e-6).log()).sum(-1).mean(-1)
                # DEBUG idle:
                # reg_loss = (soft_codes - soft_codes).sum(-1).mean(-1)
            else:
                # quantization loss
                if self.pq_detach_opt == 'none':
                    reg_loss = F.mse_loss(x_normalized, x_hat_hard, reduction='none').sum(-1)
                elif self.pq_detach_opt == 'features':
                    reg_loss = F.mse_loss(x_normalized.detach(), x_hat_hard, reduction='none').sum(-1)
                elif self.pq_detach_opt == 'codewords':
                    reg_loss = F.mse_loss(x_normalized, x_hat_hard.detach(), reduction='none').sum(-1)

            # x_hat: [bxfeat_dim], hard_codes: [bxM], soft_codes: [bxMxK]
            # x_normalized: [bxfeat_dim], reg_loss: [b]
            return x_hat, hard_codes, soft_codes, x_normalized, reg_loss
        else:
            raise RuntimeError("invalid input shape: {}".format(x.shape))


    @property
    def codebooks(self):
        return self._C_k.data, self._C_v.data

    # def save_codebooks(self, path):
    #     save_tensor(self.codebooks, path)

