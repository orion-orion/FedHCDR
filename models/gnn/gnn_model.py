# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .modules import GCN
from . import config


class GNN(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(GNN, self).__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"

        # User and item embeddings.
        # Item embeddings cannot be shared between clients, because the numbers
        # of items in each domain are different
        self.user_item_emb = nn.Embedding(
            num_users + num_items, config.emb_size)
        self.num_users, self.num_items = num_users, num_items
        # TODO: loss += args.l2_emb for regularizing embedding vectors during
        # training https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.encoder = GCN(args)

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def graph_convolution(self, all_adj):
        """Perform graph convolution.

        Args:
            all_adj: Adjacency matrix of the local user-item bipartite graph,
                    (num_users + num_items, num_users + num_items).
        """
        self.user_item_index = torch.arange(
            0, self.num_users + self.num_items, 1).to(self.device)
        # user_item_emb: (num_users + num_items, emb_size)
        user_item_emb = self.my_index_select_embedding(
            self.user_item_emb, self.user_item_index)
        self.U_and_V = self.encoder(user_item_emb, all_adj)

    def forward(self, users, items, neg_items=None):
        # `U_and_V` stores the embeddings of all users and items.
        # Here we need to select the embeddings of specific users
        # and items interacted with by the users
        # u: (batch_size, emb_size)
        u = self.my_index_select(self.U_and_V, users)
        # v: (batch_size, emb_size) in training mode,
        # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
        v = self.my_index_select(
            self.U_and_V, items + self.num_users)

        if self.training:  # Training mode
            # neg_v: (batch_size, num_neg, emb_size)
            neg_v = self.my_index_select(
                self.U_and_V, neg_items + self.num_users)
            return u, v, neg_v
        else:
            return u, v
