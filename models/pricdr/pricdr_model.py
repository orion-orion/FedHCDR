# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from . import config
from .modules import MLP, Encoder


class PriCDR(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(PriCDR, self).__init__()
        self.device = ("cuda:%s" % args.gpu if args.cuda else "cpu")

        # User embeddings
        self.user_mlp_emb = nn.Embedding(
            num_users, config.emb_size)
        self.user_mf_emb = nn.Embedding(
            num_users, config.emb_size)
        # Item embeddings cannot be shared between clients, because the numbers
        # of items in each domain are different
        self.item_mlp_emb = nn.Embedding(
            num_items, config.emb_size)
        self.item_mf_emb = nn.Embedding(
            num_items, config.emb_size)
        # TODO: loss += args.l2_emb for regularizing embedding vectors during
        # training https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.mlp = MLP(args)
        self.encoder = Encoder(args)

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

    def encode_user_embeddings(self):
        self.user_index = torch.arange(
            0, self.user_mlp_emb.num_embeddings, 1).to(self.device)
        self.item_index = torch.arange(
            0, self.item_mlp_emb.num_embeddings, 1).to(self.device)
        # user_mlp_emb, user_mf_emb: (num_users, emb_size)
        user_mlp_emb = self.my_index_select_embedding(
            self.user_mlp_emb, self.user_index)
        user_mf_emb = self.my_index_select_embedding(
            self.user_mf_emb, self.user_index)
        # item_mlp_emb, item_mf_emb: (num_items, emb_size)
        item_mlp_emb = self.my_index_select_embedding(
            self.item_mlp_emb, self.item_index)
        item_mf_emb = self.my_index_select_embedding(
            self.item_mf_emb, self.item_index)

        # U_mlp, U_mf: (num_users, emb_size)
        # V_mlp, V_mf: (num_items, emb_size)
        self.U_mlp = self.encoder(user_mlp_emb)
        self.U_mf = self.encoder(user_mf_emb)
        self.V_mlp = self.encoder(item_mlp_emb)
        self.V_mf = self.encoder(item_mf_emb)

        return self.U_mlp, self.U_mf

    def forward(self, users, items, neg_items=None, U_mlp_g=None, U_mf_g=None):
        # `U_mlp`, `U_mf` store the embeddings of all users.
        # Here we need to select the embeddings of specific users
        # u_mlp: (batch_size, emb_size)
        # u_mf: (batch_size, emb_size)
        u_mlp = self.my_index_select(self.U_mlp, users)
        u_mf = self.my_index_select(self.U_mf, users)
        u_mlp_aligned = self.my_index_select(self.U_mlp, users)
        u_mf_aligned = self.my_index_select(self.U_mf, users)
        if (U_mlp_g is not None) and (U_mf_g is not None):
            u_mlp_g = self.my_index_select(U_mlp_g, users)
            u_mf_g = self.my_index_select(U_mf_g, users)
        else:
            u_mlp_g, u_mf_g = None, None
        # `V_mlp`, `V_mf` store the embeddings of all items.
        # Here we need to select the embeddings of items interacted with by
        # specific users
        # v_mlp: (batch_size, emb_size)
        # v_mf: (batch_size, emb_size)
        v_mlp = self.my_index_select(self.V_mlp, items)
        v_mf = self.my_index_select(self.V_mf, items)

        if not self.training:  # Evaluation mode
            # (batch_size, 1, emb_size)
            u_mlp = u_mlp.view(u_mlp.size()[0], 1, -1)
            # (batch_size, 1 + num_test_neg, emb_size)
            u_mlp = u_mlp.repeat(1, v_mlp.size()[1], 1)
            # (batch_size, 1, emb_size)
            u_mf = u_mf.view(u_mf.size()[0], 1, -1)
            # (batch_size, 1 + num_test_neg, emb_size)
            u_mf = u_mf.repeat(1, v_mf.size()[1], 1)

        # The concatenated latent vector
        # mlp_vector: (batch_size, emb_size * 2) in training mode,
        # (batch_size, num_test_neg + 1, emb_size * 2) in evaluation mode
        mlp_vector = torch.cat([u_mlp, v_mlp], dim=-1)
        # Element-wise multiplication
        # mf_vector: (batch_size, emb_size) in training mode,
        # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
        mf_vector = torch.mul(u_mf, v_mf)

        # mlp_vector: (batch_size, emb_size) in training mode,
        # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
        mlp_vector = self.mlp(mlp_vector)

        if self.training:  # Training mode
            neg_v_mlp = self.my_index_select(self.V_mlp, neg_items)
            neg_v_mf = self.my_index_select(self.V_mf, neg_items)

            # (batch_size, 1, emb_size)
            u_mlp = u_mlp.view(u_mlp.size()[0], 1, -1)
            # (batch_size, num_neg, emb_size)
            u_mlp = u_mlp.repeat(1, neg_v_mlp.size()[1], 1)
            # (batch_size, 1, emb_size)
            u_mf = u_mf.view(u_mf.size()[0], 1, -1)
            # (batch_size, num_neg, emb_size)
            u_mf = u_mf.repeat(1, neg_v_mf.size()[1], 1)

            # The concatenated latent vector
            # neg_mlp_vector: (batch_size, num_neg, emb_size * 2)
            neg_mlp_vector = torch.cat([u_mlp, neg_v_mlp], dim=-1)
            # Element-wise multiplication
            # neg_mf_vector: (batch_size, num_neg, emb_size)
            neg_mf_vector = torch.mul(u_mf, neg_v_mf)

            # neg_mlp_vector: (batch_size, num_neg, emb_size)
            neg_mlp_vector = self.mlp(neg_mlp_vector)
            return mlp_vector, mf_vector, u_mlp_aligned, u_mf_aligned, \
                u_mlp_g, u_mf_g, \
                neg_mlp_vector, neg_mf_vector
        else:
            return mlp_vector, mf_vector
