# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import numpy as np


class GCNLayer(nn.Module):
    """GCN module layer.
    """

    def __init__(self, num_features, num_hidden, dropout, alpha, args):
        super(GCNLayer, self).__init__()
        self.gc1 = GraphConvolution(num_features, num_hidden, args)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        # x = self.leakyrelu(self.gc1(x, adj))
        x = self.gc1(x, adj)
        return x


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, args, bias=True):
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
