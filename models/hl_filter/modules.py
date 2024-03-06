# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GCNLayer
from . import config


class UserHyperGCN(nn.Module):
    """User hyper GCN module layer. It may be low-pass or high-pass.
    """

    def __init__(self, args, mode="low"):
        super(UserHyperGCN, self).__init__()
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
                alpha=config.leakey,
                args=args,
                mode=mode))
        self.encoder = nn.ModuleList(self.encoder)

    def forward(self, u_featues, UU_adj, read_out=False):
        learn_user = u_featues
        tmp_u_featues = u_featues
        tmp_z = u_featues.mean(axis=0)
        for layer in self.encoder:
            learn_user = F.dropout(
                learn_user, self.dropout, training=self.training)
            learn_user = layer(
                learn_user, UU_adj)
            tmp_u_featues = tmp_u_featues + learn_user
            tmp_z = tmp_z + learn_user.mean(axis=0)
        if read_out:
            # return learn_user, learn_user.mean(axis=0)
            return tmp_u_featues / (self.layer_number + 1), \
                tmp_z / (self.layer_number + 1)
        else:
            # return learn_user
            return tmp_u_featues / (self.layer_number + 1)


class ItemHyperGCN(nn.Module):
    """Item hyper GCN module layer. It may be low-pass or high-pass.
    """

    def __init__(self, args, mode="low"):
        super(ItemHyperGCN, self).__init__()
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
                alpha=config.leakey,
                args=args,
                mode=mode))
        self.encoder = nn.ModuleList(self.encoder)

    def forward(self, v_featues, VV_adj):
        learn_item = v_featues
        tmp_v_featues = v_featues
        for layer in self.encoder:
            learn_item = F.dropout(
                learn_item, self.dropout, training=self.training)
            learn_item = layer(
                learn_item, VV_adj)
            tmp_v_featues = tmp_v_featues + learn_item
        # return learn_item
        return tmp_v_featues / (self.layer_number + 1)
