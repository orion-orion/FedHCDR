# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from . import config


class GCN(nn.Module):
    """GCN module.
    """

    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.dropout = config.dropout_rate
        self.layer_number = config.num_gnn_layers

        self.encoder = []
        for i in range(self.layer_number):
            num_features = (config.emb_size if i == 0 else config.hidden_size)
            num_hidden = (config.emb_size
                          if i == self.layer_number - 1
                          else config.hidden_size)
            self.encoder.append(GCNLayer(
                num_features=num_features,
                num_hidden=num_hidden,
                dropout=config.dropout_rate,
                alpha=config.leakey))
        self.encoder = nn.ModuleList(self.encoder)

    def forward(self, fea, adj):
        learn_fea = fea
        tmp_fea = fea
        for layer in self.encoder:
            learn_fea = F.dropout(learn_fea, self.dropout,
                                  training=self.training)
            learn_fea = layer(learn_fea, adj)
            tmp_fea = tmp_fea + learn_fea
        # return learn_user
        return tmp_fea / (self.layer_number + 1)


class GCNLayer(nn.Module):
    """GCN module layer.
    """

    def __init__(self, num_features, num_hidden, dropout, alpha):
        super(GCNLayer, self).__init__()
        self.gc1 = GraphConvolution(num_features, num_hidden)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        # x = self.leakyrelu(self.gc1(x, adj))
        x = self.gc1(x, adj)
        return x


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        # self.weight = self.glorot_init(in_features, out_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * \
            2 * init_range - init_range
        return nn.Parameter(initial / 2)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + " (" \
            + str(self.in_features) + " -> " \
            + str(self.out_features) + ")"
