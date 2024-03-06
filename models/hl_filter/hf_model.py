# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .modules import UserHyperGCN, ItemHyperGCN
from . import config


class HF(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(HF, self).__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"

        # User embeddings
        self.user_emb = nn.Embedding(
            num_users, config.emb_size)
        # Item embeddings cannot be shared between clients, because the numbers
        # of items in each domain are different
        self.item_emb = nn.Embedding(
            num_items, config.emb_size)
        # TODO: loss += args.l2_emb for regularizing embedding vectors during
        # training https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.user_encoder = UserHyperGCN(args)
        self.item_encoder = ItemHyperGCN(args)

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

    def graph_convolution(self, UU_adj, VV_adj):
        """Perform graph convolution.

        Args:
            UU_adj: User hypergraph adjacency matrix, (num_users, num_users).
            VV_adj: Item hypergraph adjacency matrix, (num_items, num_items).
        """
        self.user_index = torch.arange(
            0, self.user_emb.num_embeddings, 1).to(self.device)
        self.item_index = torch.arange(
            0, self.item_emb.num_embeddings, 1).to(self.device)
        # user_emb: (num_users, emb_size)
        user_emb = self.my_index_select_embedding(
            self.user_emb, self.user_index)
        # item_emb: (num_items, emb_size)
        item_emb = self.my_index_select_embedding(
            self.item_emb, self.item_index)

        # U: (num_users, emb_size)
        # V: (num_items, emb_size)
        self.U = self.user_encoder(user_emb, UU_adj)
        self.V = self.item_encoder(item_emb, VV_adj)

    def forward(self, users, items, neg_items=None):
        # `U` stores the embeddings of all users.
        # Here we need to select the embeddings of specific users
        # u: (batch_size, emb_size)
        u = self.my_index_select(self.U, users)
        # `V` stores the embeddings of all items.
        # Here we need to select the embeddings of items interacted with by specific
        # users
        # v: (batch_size, emb_size) in training mode,
        # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
        v = self.my_index_select(
            self.V, items)

        if self.training and (neg_items is not None):  # Training mode
            # (batch_size, num_neg, emb_size)
            neg_v = self.my_index_select(
                self.V, neg_items)
            return u, v, neg_v
        else:
            return u, v
