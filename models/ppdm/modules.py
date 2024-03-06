# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from . import config


class PGCN(nn.Module):
    """PGCN module.
    """

    def __init__(self, args):
        super(PGCN, self).__init__()
        self.args = args
        self.dropout = config.dropout_rate
        self.layer_number = config.num_gnn_layers

        self.mu_encoder = []
        for i in range(self.layer_number):
            num_features = (config.emb_size if i == 0 else config.hidden_size)
            num_hidden = (config.emb_size
                          if i == self.layer_number - 1
                          else config.hidden_size)
            self.mu_encoder.append(GCNLayer(
                num_features=num_features,
                num_hidden=num_hidden,
                dropout=config.dropout_rate,
                alpha=config.leakey))
        self.mu_encoder = nn.ModuleList(self.mu_encoder)

        self.sigma_encoder = []
        for i in range(self.layer_number):
            num_features = (config.emb_size if i == 0 else config.hidden_size)
            num_hidden = (config.emb_size
                          if i == self.layer_number - 1
                          else config.hidden_size)
            self.sigma_encoder.append(GCNLayer(
                num_features=num_features,
                num_hidden=num_hidden,
                dropout=config.dropout_rate,
                alpha=config.leakey))
        self.sigma_encoder = nn.ModuleList(self.sigma_encoder)

    def forward(self, fea, adj):
        mu_learn_fea = fea
        mu_tmp_fea = fea
        for mu_layer in self.mu_encoder:
            mu_learn_fea = F.dropout(mu_learn_fea, self.dropout,
                                     training=self.training)
            mu_learn_fea = mu_layer(mu_learn_fea, adj)
            mu_tmp_fea = mu_tmp_fea + mu_learn_fea

        sigma_learn_fea = fea
        sigma_tmp_fea = fea
        for sigma_layer in self.sigma_encoder:
            sigma_learn_fea = F.dropout(sigma_learn_fea, self.dropout,
                                        training=self.training)
            sigma_learn_fea = sigma_layer(sigma_learn_fea, adj)
            sigma_tmp_fea = sigma_tmp_fea + sigma_learn_fea

        # return mu_learn_fea, sigma_learn_fea
        return mu_tmp_fea / (self.layer_number + 1), \
            sigma_tmp_fea / (self.layer_number + 1)


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
