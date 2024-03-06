# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .modules import PGCN
from . import config


class PPDM(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(PPDM, self).__init__()
        self.device = ("cuda:%s" % args.gpu if args.cuda else "cpu")

        # User and item embeddings
        # Item embeddings cannot be shared between clients, because the numbers
        # of items in each domain are different
        self.user_item_emb = nn.Embedding(
            num_items + num_users, config.emb_size)
        self.num_users, self.num_items = num_users, num_items
        # TODO: loss += args.l2_emb for regularizing embedding vectors during
        # training https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.encoder = PGCN(args)

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
            0, self.num_items + self.num_users, 1).to(self.device)
        # user_item_emb: (num_users + num_items, emb_size)
        user_item_emb = self.my_index_select_embedding(
            self.user_item_emb, self.user_item_index)
        self.U_and_V_mu, self.U_and_V_sigma \
            = self.encoder(user_item_emb, all_adj)
        self.U_mu = self.U_and_V_mu[: self.num_users]
        self.U_sigma = self.U_and_V_sigma[: self.num_users]
        return self.U_mu, self.U_sigma

    def forward(self, users, items, neg_items=None,
                U_mu_g=None, U_sigma_g=None):
        # `U_and_V_mu`, `U_and_V_sigma` stores the embeddings of all users and
        # items. Here we need to select the embeddings of specific users and
        # items interacted with by the users
        # u_mu, u_sigam: (batch_size, emb_size)
        u_mu = self.my_index_select(self.U_and_V_mu, users)
        u_sigma = self.my_index_select(self.U_and_V_sigma, users)
        if (U_mu_g is not None) and (U_sigma_g is not None):
            # u_mu_g, u_sigam_g: (batch_size, emb_size)
            u_mu_g = self.my_index_select(U_mu_g, users)
            u_sigma_g = self.my_index_select(U_sigma_g, users)
        else:
            u_mu_g, u_sigma_g = None, None
        # v_mu, v_sigma: (batch_size, emb_size) in training mode,
        # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
        v_mu = self.my_index_select(
            self.U_and_V_mu, items + self.num_users)
        v_sigma = self.my_index_select(
            self.U_and_V_sigma, items + self.num_users)

        if self.training:  # Training mode
            # neg_v_mu, neg_v_sigma: (batch_size, num_neg, emb_size)
            neg_v_mu = self.my_index_select(
                self.U_and_V_mu, neg_items + self.num_users)
            neg_v_sigma = self.my_index_select(
                self.U_and_V_sigma, neg_items + self.num_users)
            return u_mu, u_sigma, v_mu, v_sigma, \
                u_mu_g, u_sigma_g, \
                neg_v_mu, neg_v_sigma
        else:
            return u_mu, u_sigma, v_mu, v_sigma
