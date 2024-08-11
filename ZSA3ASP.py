"""
AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import random
from typing import Union
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
# from torchstat import stat
from thop import profile
import time

# from main import get_model
import json
import torch.nn.init as init
from torch.autograd import Function
from typing import Dict, List, Union
from importlib import import_module

class spec_average(nn.Module):
    """
    :param lfcc: feature tensor. in our case was of dimensions 60X450/  70, 64472
    :param frequency_masking_para: max number of bins to mask in frequency
    :param time_masking_para: max number of consecutive steps to mask in time
    :param frequency_mask_num: number of masks in frequency
    :param time_mask_num: number of masks in time
    :param specavg: True  = SpecAverage. False = SpecAugment.
    :return: masked feature
    "paper:A-study-on-data-aug..."
    """
    def __init__(self, frequency_masking_para=12, time_masking_para=10745, frequency_mask_num=1, time_mask_num=1, specavg = True):
        super(spec_average, self).__init__()

        self.frequency_masking_para = frequency_masking_para
        self.time_masking_para = time_masking_para
        self.frequency_mask_num = frequency_mask_num
        self.time_mask_num = time_mask_num
        self.specavg = specavg

    def forward(self, inputs):
        if self.specavg:
            avg = inputs.mean()
            val = avg
        else: # Don't put average, put 0
            val = 0
        # print("spec_uts", inputs)
        v = inputs.shape[1]

        tau = inputs.shape[2]

        # Step 1 : Frequency masking
        for i in range(self.frequency_mask_num):
            f = np.random.uniform(low=0.0, high=self.frequency_masking_para)
            f = int(f)
            f0 = random.randint(0, v-f)
            inputs[:, f0:f0+f, :] = val

        # Step 2 : Time masking
        for i in range(self.time_mask_num):
            t = np.random.uniform(low=0.0, high=self.time_masking_para)
            t = int(t)
            # print("t:",t)
            t0 = random.randint(0, tau-t)
            inputs[:, :, t0:t0+t] = val

        return inputs

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()

        # self.output_size = output_size
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        self.mean_only = mean_only

        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):
        # print('inputs.shape:', inputs.shape)   inputs.shape: torch.Size([64, 32, 256])
        batch_size = inputs.size(0)
        inputs2 = self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1)
        # print('inputs2.shape', inputs2.shape)   inputs2.shape torch.Size([64, 256, 1])
        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))
        # print('weights.shape:', weights.shape)
        # weights.shape torch.Size([64, 32, 1])
        if inputs.size(0) == 1:
            attentions = F.softmax(torch.tanh(weights), dim=1)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            # print(torch.tanh(weights.squeeze()).shape)
            attentions = F.softmax(torch.tanh(weights.squeeze()), dim=1)
            # print('attentions.shape:', attentions.shape)   torch.Size([64, 94])
            attentions2 = attentions.unsqueeze(2).expand_as(inputs)
            # print('attentions2.shape:', attentions2.shape)   torch.Size([64, 94, 256])
            weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))
            # print('weighted.shape:', weighted.shape)
            # weighted.shape: torch.Size([64, 32, 256])
            # weighted.shape: torch.Size([4, 23, 64])
        # return weighted
        if self.mean_only:
            return weighted.sum(1)
        else:
            noise = 1e-5 * torch.randn(weighted.size())
            # print(weighted.sum(1).shape)   torch.Size([4, 64])
            if inputs.is_cuda:
                noise = noise.to(inputs.device)
            avg_repr, std_repr = weighted.sum(1), (weighted + noise).std(1)
            # print("avg::",avg_repr.shape,std_repr.shape)
            # torch.Size([64, 256]) torch.Size([64, 256])
            representations = torch.cat((avg_repr, std_repr), 1)

            return representations


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)

        x = torch.cat([x1, x2], dim=1)
        # print('x.shape:', x.shape)
        # x.shape: torch.Size([4, 31, 64])
        # x.shape: torch.Size([4, 15, 32])
        # x.shape: torch.Size([4, 31, 64])
        # x.shape: torch.Size([4, 15, 32])

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)

        # directional edge for master node
        master = self._update_master(x, master)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        # print('x1.shape:', x1.shape)
        # print('x2.shape:', x2.shape)
        # print('master.shape:', master.shape)
        return x1, x2, master

        # x1.shape: torch.Size([4, 20, 32])
        # x2.shape: torch.Size([4, 11, 32])
        # master.shape: torch.Size([4, 1, 32])
        # x1.shape: torch.Size([4, 10, 32])
        # x2.shape: torch.Size([4, 5, 32])
        # master.shape: torch.Size([4, 1, 32])
        # x1.shape: torch.Size([4, 20, 32])
        # x2.shape: torch.Size([4, 11, 32])
        # master.shape: torch.Size([4, 1, 32])
        # x1.shape: torch.Size([4, 10, 32])
        # x2.shape: torch.Size([4, 5, 32])
        # master.shape: torch.Size([4, 1, 32])



    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        # att_map = torch.matmul(att_map, self.att_weight12)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)

        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 sample_rate=16000,
                 in_channels=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 groups=1,
                 mask=False):
        super().__init__()
        if in_channels != 1:

            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * \
                np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * \
                np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))
        # self.mp = nn.MaxPool2d((1, 4))
        # self.ap = nn.AvgPool2d((1, 4))
        # out.shape: torch.Size([4, 64, 23, 20])
        # self.mp = nn.MaxPool2d((1, 5))
        # out.shape: torch.Size([4, 64, 23, 34])

    def forward(self, x):
        identity = x
        # print('x:', x.shape)
        # torch.Size([4, 1, 23, 21490])
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)
        # out = self.conv1(out)   #注释的这里是加的，但是没采用

        # print('conv1 out:', out.shape)
        # torch.Size([4, 32, 24, 21490])
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        # print('conv2 out.shape:', out.shape)
        # torch.Size([4, 32, 23, 21490])
        if self.downsample:
            identity = self.conv_downsample(identity)
            # print("indentity .shape:", identity.shape)
        out += identity
        # out2 = self.ap(out)
        out = self.mp(out)
        # out = 0.5 * (out+out2)
        # print('out.shape:', out.shape)
        # out.shape: torch.Size([4, 32, 23, 7163])
        # out.shape: torch.Size([4, 32, 23, 2387])
        # out.shape: torch.Size([4, 64, 23, 795])
        # out.shape: torch.Size([4, 64, 23, 265])
        # out.shape: torch.Size([4, 64, 23, 88])
        # out.shape: torch.Size([4, 64, 23, 29])
        return out


class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()

        self.d_args = d_args
        filts = d_args["filts"]        #"filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        gat_dims = d_args["gat_dims"]   #[64, 32],
        pool_ratios = d_args["pool_ratios"]   #[0.5, 0.7, 0.5, 0.5],
        temperatures = d_args["temperatures"]
        self.conv_time = CONV(out_channels=filts[0],              #70
                              kernel_size=d_args["first_conv"],   #128
                              in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)

        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)
        self.augmask = spec_average(12, 10745, 1, 1, True)
        self.attention = SelfAttention(32)
        self.attention_asp = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4]))
        )

        self.pos_S = nn.Parameter(torch.randn(1, 23, 128))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))  #[1,1,64]
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(128,
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(128,
                                               gat_dims[0],
                                               temperature=temperatures[1])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])   #(64,32,100.0)
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])   #(32,32,100.0)

        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])   #(64,32,100.0)
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])   #(32,32,100.0)

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(9 * gat_dims[1], 2)    #288-->2

    def forward(self, x, augmask=False):

        x = x.unsqueeze(1)
        # print(x.shape)torch.Size([4, 1, 64600])
        x = self.conv_time(x, mask=False)
        # print("conv_x.shape:", x.shape)
        # conv_x.shape: torch.Size([4, 70, 64472])
        if augmask:
            x = self.augmask(x)
        # print("shape:", x.shape)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        # print("xmax_pool2d.shape:", x.shape)   xmax_pool2d.shape: torch.Size([4, 1, 23, 21490])
        x = self.first_bn(x)
        x = self.selu(x)

        # get embeddings using encoder
        # (#bs, #filt, #spec, #seq)
        e = self.encoder(x)     # [#bs, C(64), F(23), T(29)]
        # ASP
        ts = e.size()[2]
        tt = e.size()[3]
        global_e = torch.cat((e, torch.mean(e, dim=(2,3), keepdim=True).repeat(1, 1, ts, tt),
                              torch.sqrt(torch.var(e, dim=(2,3), keepdim=True).clamp(min=1e-4)).repeat(1, 1, ts, tt)), dim=1)
        w = self.attention_asp(e)
        # ASP_S
        w1 = F.softmax(w, dim=3)
        # print("w1.shape:", w1.shape)  w1.shape: torch.Size([4, 64, 23, 29])
        mu_s = torch.sum(e * w1, dim=3)
        sg_s = torch.sqrt((torch.sum((e ** 2) * w1, dim=3) - mu_s ** 2).clamp(min=1e-4))
        e_S = torch.cat((mu_s, sg_s), 1)
        e_S = torch.abs(e_S)
        # print("e_S.shape:", e_S.shape)  torch.Size([4, 128, 23])

        # spectral GAT (GAT-S)
        # e_S, _ = torch.max(torch.abs(e), dim=3)  # max along time  #[#bs, C(64), F(23)]
        e_S = e_S.transpose(1, 2) + self.pos_S
        # print("e_S.shape:", e_S.shape)   e_S.shape: torch.Size([4, 23, 128])

        gat_S = self.GAT_layer_S(e_S)
        # print("gat_S.shape:", gat_S.shape)    待定gat_S.shape: torch.Size([4, 23, 64])
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)
        # print("out_S.shape:", out_S.shape)    out_S.shape: torch.Size([4, 11, 64])

        # ASP_T
        w2 = F.softmax(w, dim=2)
        # print("w2.shape:", w2.shape)  torch.Size([4, 64, 23, 29])
        # torch.Size([4, 64, 23, 29])
        mu_t = torch.sum(e * w2, dim=2)
        sg_t = torch.sqrt((torch.sum((e ** 2) * w2, dim=2) - mu_t ** 2).clamp(min=1e-4))
        e_T = torch.cat((mu_t, sg_t), 1)
        e_T = torch.abs(e_T)
        # print("e_T.shape:", e_T.shape)  # torch.Size([4, 128, 29])

        # temporal GAT (GAT-T)
        # e_T, _ = torch.max(torch.abs(e), dim=2)  # max along freq   #[#bs, C(64), T(29)]
        e_T = e_T.transpose(1, 2)
        # torch.Size([4, 29, 128])

        gat_T = self.GAT_layer_T(e_T)
        # print("gat_T.shape:", gat_T.shape)   gat_T.shape: torch.Size([4, 29, 64])
        out_T = self.pool_T(gat_T)
        # print("out_T.shape:", out_T.shape)  out_T.shape: torch.Size([4, 20, 64])

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)      #[bs,1,64]
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        # T1.shape: torch.Size([4, 20, 32])
        # S1.shape: torch.Size([4, 11, 32])
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug


        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)
        # master1.shape: torch.Size([4, 1, 32])

        # ZSA3
        out_T1 = self.attention(out_T1)
        out_T2 = self.attention(out_T2)
        out_S1 = self.attention(out_S1)
        out_S2 = self.attention(out_S2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)
        # print("out_T.shape:", out_T.shape)  torch.Size([4, 64])
        # print("out_S.shape:", out_S.shape)   torch.Size([4, 64])
        # print("master.shape:", master.shape)   master.shape: torch.Size([4, 1, 32])

        T_max = torch.abs(out_T)
        out_Tzsa3 = out_T.unsqueeze(1)
        T_avg = torch.mean(out_Tzsa3, dim=1)
        # print("T_max.shape:", T_max.shape)
        # print("T_avg.shape:", T_avg.shape)

        S_max = torch.abs(out_S)
        out_Szsa3 = out_S.unsqueeze(1)
        S_avg = torch.mean(out_Szsa3, dim=1)
        # print("S_max.shape:", S_max.shape)
        # print("S_avg.shape:", S_avg.shape)

        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        # print("last_hidden:", last_hidden.shape)   torch.Size([4, 160])

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return last_hidden, output


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class ChannelClassifier(nn.Module):
    def __init__(self, enc_dim, nclasses, lambda_=0.05, ADV=True):
        super(ChannelClassifier, self).__init__()
        self.adv = ADV
        if self.adv:
            self.grl = GradientReversal(lambda_)
        self.classifier = nn.Sequential(nn.Linear(enc_dim, enc_dim // 2),
                                        nn.Dropout(0.3),
                                        nn.ReLU(),
                                        nn.Linear(enc_dim // 2, nclasses),
                                        nn.ReLU())

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        if self.adv:
            x = self.grl(x)
            # print("在这里")
        return self.classifier(x)





if __name__ == "__main__":
    torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    with open('../config/AASIST.conf', "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    classifier = ChannelClassifier(160, 4, lambda_=0.05, ADV=True).to(device)

    def get_model(model_config: Dict, device: torch.device):
        """Define DNN model architecture"""
        module = import_module("models.{}".format(model_config["architecture"]))
        _model = getattr(module, "Model")
        model = _model(model_config).to(device)
        nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
        print("no. model params:{}".format(nb_params))
        return model

    model1 = get_model(model_config, device)

    x = torch.randn(1, 64600).to(device)

    flops, params = profile(model1, inputs=(x,))
    print(f"FLOPs: {flops}, Params: {params}")
    # start = time.time()
    # print(start)
    last_hidden, output = model1(x)
    # print("lasthidden,shape:",last_hidden.shape)
    # classifier_out = classifier(last_hidden)
    # # # output = F.softmax(output, dim=1)
    # # end = time.time()
    # print(classifier_out)
    print(output.shape)


    # print(end)
    # print(end-start)