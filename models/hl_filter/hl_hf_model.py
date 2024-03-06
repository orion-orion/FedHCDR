# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from . import config
from .modules import UserHyperGCN, ItemHyperGCN


class HL_HF(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(HL_HF, self).__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.hi_model = HighFilter(num_users, num_items, args)
        self.lo_model = LowFilter(num_users, num_items, args)

    def forward_hi(self, users, items, neg_users=None, neg_items=None):
        return self.hi_model.forward(users, items, neg_users, neg_items)

    def forward_lo(self, users, items=None,
                   U_g=None,
                   neg_users=None, neg_items=None):
        return self.lo_model.forward(users, items, U_g, neg_users, neg_items)

    def graph_convolution_hi(self, UU_adj, VV_adj):
        self.hi_model.graph_convolution(UU_adj, VV_adj)

    def graph_convolution_lo(self, UU_adj, VV_adj,
                             M=None, perturb_UU_adj=None):
        return self.lo_model.graph_convolution(UU_adj, VV_adj,
                                               M, perturb_UU_adj)


class HighFilter(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(HighFilter, self).__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"

        # User embeddings
        self.user_emb_e = nn.Embedding(
            num_users, config.emb_size)
        # Item embeddings cannot be shared between clients, because the numbers
        # of items in each domain are different
        self.item_emb_e = nn.Embedding(
            num_items, config.emb_size)
        # TODO: loss += args.l2_emb for regularizing embedding vectors during
        # training https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.user_hyper_gcn_hi = UserHyperGCN(args, mode="high")
        self.item_hyper_gcn_hi = ItemHyperGCN(args, mode="high")

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
            0, self.user_emb_e.num_embeddings, 1).to(self.device)
        self.item_index = torch.arange(
            0, self.item_emb_e.num_embeddings, 1).to(self.device)
        # user_emb_e: (num_users, emb_size)
        user_emb_e = self.my_index_select_embedding(
            self.user_emb_e, self.user_index)
        # item_emb_e: (num_items, emb_size)
        item_emb_e = self.my_index_select_embedding(
            self.item_emb_e, self.item_index)

        # U_e: (num_users, emb_size)
        # V_e: (num_items, emb_size)
        self.U_e = self.user_hyper_gcn_hi(user_emb_e, UU_adj)
        self.V_e = self.item_hyper_gcn_hi(item_emb_e, VV_adj)

    def forward(self, users, items, neg_users=None, neg_items=None):
        # `U_e` stores the embeddings of all users.
        # Here we need to select the embeddings of specific users
        # u_e: (batch_size, emb_size)
        u_e = self.my_index_select(self.U_e, users)
        # `V_e` stores the embeddings of all items.
        # Here we need to select the embeddings of items interacted with by
        # specific users
        # v_e: (batch_size, emb_size) in training mode,
        # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
        v_e = self.my_index_select(
            self.V_e, items)

        if self.training and (neg_users is not None):  # Training mode
            # (batch_size, emb_size)
            neg_u_e = self.my_index_select(
                self.U_e, neg_users)
            # (batch_size, num_neg, emb_size)
            neg_v_e = self.my_index_select(
                self.V_e, neg_items)
            return u_e, neg_u_e, v_e, neg_v_e
        else:
            return u_e, v_e


class LowFilter(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(LowFilter, self).__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"

        # User embeddings
        self.user_emb_s = nn.Embedding(
            num_users, config.emb_size)
        # Item embeddings cannot be shared between clients, because the numbers
        # of items in each domain are different
        self.item_emb_s = nn.Embedding(
            num_items, config.emb_size)
        # TODO: loss += args.l2_emb for regularizing embedding vectors during
        # training https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.user_emb_rw = torch.nn.Linear(args.n_rw, config.emb_size)
        self.user_hyper_gcn_lo = UserHyperGCN(args, mode="low")
        self.item_hyper_gcn_lo = ItemHyperGCN(args, mode="low")

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

    def graph_convolution(self, UU_adj, VV_adj, M=None, perturb_UU_adj=None):
        """Perform graph convolution.

        Args:
            UU_adj: User hypergraph adjacency matrix, (num_users, num_users).
            VV_adj: Item hypergraph adjacency matrix, (num_items, num_items).
        """
        self.user_index = torch.arange(
            0, self.user_emb_s.num_embeddings, 1).to(self.device)
        self.item_index = torch.arange(
            0, self.item_emb_s.num_embeddings, 1).to(self.device)
        # If use embedding initialization strategy
        if M is not None:
            # user_emb_s: (num_users, emb_size)
            user_emb_s = self.user_emb_rw(M)
        else:
            user_emb_s = self.my_index_select_embedding(
                self.user_emb_s, self.user_index)
        # item_emb_s: (num_items, emb_size)
        item_emb_s = self.my_index_select_embedding(
            self.item_emb_s, self.item_index)

        # U_s: (num_users, emb_size)
        # z_s: (emb_size, )
        # V_s: (num_items, emb_size)
        self.U_s, self.z_s = self.user_hyper_gcn_lo(
            user_emb_s, UU_adj, read_out=True)
        self.V_s = self.item_hyper_gcn_lo(item_emb_s, VV_adj)

        if perturb_UU_adj is not None:
            # aug_U_s: (num_users, emb_size)
            # aug_z_s: (emb_size, )
            self.aug_U_s, self.aug_z_s = self.user_hyper_gcn_lo(
                user_emb_s, perturb_UU_adj, read_out=True)
            return self.U_s, self.z_s, self.aug_z_s
        else:
            return self.U_s

    def forward(self, users, items=None,
                U_g=None,
                neg_users=None, neg_items=None):
        if items is None:
            if U_g is not None:
                # `U_g` stores the global embeddings of all users.
                # Here we need to select the global embeddings of specific
                # users
                # u_g: (batch_size, emb_size)
                u_g = self.my_index_select(U_g, users)
                return u_g
            else:
                return None
        else:
            # u_s: (batch_size, emb_size)
            u_s = self.my_index_select(self.U_s, users)
            # `V_s` stores the embeddings of all items.
            # Here we need to select the embeddings of items interacted with
            # by specific users
            # v_s: (batch_size, emb_size) in training mode,
            # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
            v_s = self.my_index_select(
                self.V_s, items)
            # Training mode
            if self.training and (neg_users is not None) \
                    and (neg_items is not None):
                # aug_u_s: (batch_size, emb_size)
                # neg_u_s: (batch_size, emb_size)
                # neg_v_s: (batch_size, num_neg, emb_size)
                aug_u_s = self.my_index_select(self.aug_U_s, users)
                neg_u_s = self.my_index_select(
                    self.U_s, neg_users)
                neg_v_s = self.my_index_select(
                    self.V_s, neg_items)
                return u_s, aug_u_s, neg_u_s, v_s, neg_v_s
            else:
                return u_s, v_s
