# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hl_filter.hl_hf_model import HL_HF
from models.hl_filter.hf_model import HF
from models.dhcf.dhcf_model import DHCF
from models.mf.mf_model import NeuMF
from models.gnn.gnn_model import GNN
from models.pricdr.pricdr_model import PriCDR
from models.p2fcdr.p2fcdr_model import P2FCDR
from models.ppdm.ppdm_model import PPDM
from utils import train_utils
from losses import HingeLoss, JSDLoss, Discriminator


class Trainer(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def train_batch(self, *args, **kwargs):
        raise NotImplementedError

    def test_batch(self, *args, **kwargs):
        raise NotImplementedError

    def update_lr(self, new_lr):
        if self.method == "FedHCDR":
            train_utils.change_lr(self.hi_optimizer, new_lr)
            train_utils.change_lr(self.lo_optimizer, new_lr)
        else:
            train_utils.change_lr(self.optimizer, new_lr)


class ModelTrainer(Trainer):
    def __init__(self, args, num_users, num_items):
        self.args = args
        self.method = args.method
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        if self.method == "FedHCDR":
            self.model = HL_HF(num_users, num_items, args).to(self.device)
            # Here we set `self.U_s[:], self.U_g = [None], [None]` so that
            # we can use `self.U_s[:] = ...`, `self.U_g[:] = ...` to modify
            # them later.
            # Note that if we set `self.U_s, self.U_g = None, None`,
            # then `self.U_s = obj` / `self.U_g = obj` will just refer to a
            # new object `obj`, rather than modify `self.U_s` / `self.U_g`
            # itself
            self.U_s, self.U_g = [None], [None]
        elif "HF" in self.method:
            self.model = HF(num_users, num_items, args).to(self.device)
        elif "DHCF" in self.method:
            self.model = DHCF(num_users, num_items, args).to(self.device)
        elif "MF" in self.method:
            self.model = NeuMF(num_users, num_items, args).to(self.device)
        elif "GNN" in self.method:
            self.model = GNN(num_users, num_items, args).to(self.device)
        elif "PriCDR" in self.method:
            self.model = PriCDR(num_users, num_items, args).to(self.device)
            self.U_mlp, self.U_mf, self.U_mlp_g, self.U_mf_g \
                = [None], [None], [None], [None]
        elif "P2FCDR" in self.method:
            self.model = P2FCDR(num_users, num_items, args).to(self.device)
            self.U_mlp, self.U_mf, self.U_mlp_g, self.U_mf_g \
                = [None], [None], [None], [None]
        elif "PPDM" in self.method:
            self.model = PPDM(num_users, num_items, args).to(self.device)
            self.U_mu, self.U_sigma, self.U_mu_g, self.U_sigma_g \
                = [None], [None], [None], [None]
            self.V_mu, self.V_sigma, self.V_mu_g, self.V_sigma_g \
                = [None], [None], [None], [None]
        if ("MF" in self.method) or ("PriCDR" in self.method) or \
                ("P2FCDR" in self.method):
            from models.mf import config
            self.discri = Discriminator(config.emb_size).to(self.device)
        else:
            from models.hl_filter import config
            self.discri = Discriminator(config.emb_size).to(self.device)

        self.bce_criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.jsd_criterion = JSDLoss().to(self.device)
        self.hinge_criterion = HingeLoss(margin=0.3).to(self.device)

        if args.method == "FedHCDR":
            self.hi_params = list(self.model.hi_model.parameters())
            self.lo_params = list(self.model.lo_model.parameters())
            self.hi_params += list(self.discri.parameters())
            self.lo_params += list(self.discri.parameters())
        elif ("DHCF" in args.method) or ("HF" in args.method):
            self.params = list(self.model.parameters()) + \
                list(self.discri.parameters())
        else:
            self.params = list(self.model.parameters()) + \
                list(self.discri.parameters())

        if args.method == "FedHCDR":
            self.hi_optimizer = train_utils.get_optimizer(
                args.optimizer, self.hi_params, args.lr)
            self.lo_optimizer = train_utils.get_optimizer(
                args.optimizer, self.lo_params, args.lr)
        else:
            self.optimizer = train_utils.get_optimizer(
                args.optimizer, self.params, args.lr)

        self.step = 0

    def train_batch(self, users, interactions, round, args,
                    UU_adj=None, VV_adj=None, M=None, perturb_UU_adj=None,
                    all_adj=None,
                    zeta=None, tilde_u_mu=None, tilde_u_sigma=None,
                    global_params=None):
        """Trains the model for one batch.

        Args:
            users: Input user IDs.
            interactions: Input user interactions.
            round: Global training round.
            args: Other arguments for training.
            UU_adj: User-item incidence matrix.
            perturb_UU_adj: Parameter in FedHCDR method (ours).
            VV_adj: User-item incidence matrix.
            M: Parameter in FedHCDR method (ours).
            all_adj: Adjacency matrix of the local user-item bipartite graph.    
            zeta: Parameter in FedPPDM++ method.
            tilde_u_mu: Parameter in FedPPDM++ method.
            tilde_u_sigma: Parameter in FedPPDM++ method.
            global_params: Global model parameters used in `FedProx` method.
        """
        if self.method != "FedHCDR":
            self.optimizer.zero_grad()

        if self.method == "FedHCDR":
            # Here the items are first sent to GNN for convolution, and then
            # the resulting embeddings are sent to the rating module.
            # Note that each batch must be convolved once, and the
            # item_embeddings input to the convolution layer are updated from
            # the previous batch.
            self.model.graph_convolution_hi(UU_adj, VV_adj)
            # Here `M` is used for initializing the embeddings of low-pass
            # user hypergraph filter,
            # `perturb_UU_adj` is used for computing hypergraph contrastive
            # loss
            self.U_s[0], z_s, aug_z_s = self.model.graph_convolution_lo(
                UU_adj, VV_adj, M, perturb_UU_adj)
        elif "HF" in self.method:
            self.model.graph_convolution(UU_adj, VV_adj)
        elif "DHCF" in self.method:
            self.model.graph_convolution(UU_adj, VV_adj)
        elif "GNN" in self.method:
            self.model.graph_convolution(all_adj)
        elif "PriCDR" in self.method:
            self.U_mlp[0], self.U_mf[0] = self.model.encode_user_embeddings()
        elif "P2FCDR" in self.method:
            self.U_mlp[0], self.U_mf[0] = self.model.get_user_embeddings()
        elif "PPDM" in self.method:
            self.U_mu[0], self.U_sigma[0] = self.model.graph_convolution(
                all_adj)

        users = torch.LongTensor(users).to(self.device)
        interactions = [torch.LongTensor(x).to(
            self.device) for x in interactions]

        if self.method == "FedHCDR":
            # items: (batch_size, )
            # neg_items:  (batch_size, num_neg),
            # neg_users: (batch_size, ),
            items, neg_items, neg_users = interactions

            self.hi_optimizer.zero_grad()
            self.lo_optimizer.zero_grad()

            # u_g: (batch_size, emb_size)
            # Note that `u_g` is `None` in the first round
            u_g = self.model.forward_lo(users, U_g=self.U_g[0])
            # u_e: (batch_size, emb_size)
            # neg_u_e: (batch_size, emb_size)
            # v_e: (batch_size, emb_size)
            # neg_v_e: (batch_size, num_neg, emb_size)
            u_e, neg_u_e, v_e, neg_v_e = self.model.forward_hi(
                users, items, neg_users=neg_users, neg_items=neg_items)

            hi_loss = self.hi_loss_fn(u_e, u_g, v_e, neg_v_e, neg_u_e, round)

            hi_loss.backward()
            self.hi_optimizer.step()

            u_s, aug_u_s, neg_u_s, v_s, neg_v_s = self.model.\
                forward_lo(users, items, neg_users=neg_users,
                           neg_items=neg_items)
            lo_loss = self.lo_loss_fn(u_s, aug_u_s, u_e, v_s, neg_v_s,
                                      neg_u_s, z_s, aug_z_s, round)

            lo_loss.backward()
            self.lo_optimizer.step()

            self.step += 1
            return (hi_loss.item() + lo_loss.item())/2

        elif "HF" in self.method:
            items, neg_items = interactions
            # u: (batch_size, emb_size)
            # v: (batch_size, emb_size)
            # neg_v: (batch_size, num_neg, emb_size)
            u, v, neg_v = self.model(users, items, neg_items)
            loss = self.hf_loss_fn(u, v, neg_v)
            if ("Fed" in self.method) and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "DHCF" in self.method:
            items, neg_items = interactions
            # u: (batch_size, emb_size)
            # v: (batch_size, emb_size)
            # neg_v: (batch_size, num_neg, emb_size)
            u, v, neg_v = self.model(users, items, neg_items)
            loss = self.dhcf_loss_fn(u, v, neg_v)
            if ("Fed" in self.method) and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "MF" in self.method:
            items, neg_items = interactions
            # mlp_vector, mf_vector: (batch_size, emb_size)
            # neg_mlp_vector, neg_mf_vector: (batch_size, num_neg, emb_size)
            mlp_vector, mf_vector, neg_mlp_vector, neg_mf_vector = self.model(
                users, items, neg_items)
            loss = self.mf_loss_fn(mlp_vector, mf_vector,
                                   neg_mlp_vector, neg_mf_vector)
            if ("Fed" in self.method) and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "GNN" in self.method:
            items, neg_items = interactions
            # u: (batch_size, emb_size)
            # v: (batch_size, emb_size)
            # neg_v: (batch_size, num_neg, emb_size)
            u, v, neg_v = self.model(users, items, neg_items)
            loss = self.gnn_loss_fn(u, v, neg_v)
            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "PriCDR" in self.method:
            items, neg_items = interactions
            # mlp_vector, mf_vector: (batch_size, emb_size)
            # u_mlp, u_mf, u_mlp_g, u_mf_g: (batch_size, emb_size)
            # neg_mlp_vector, neg_mf_vector: (batch_size, num_neg, emb_size)
            mlp_vector, mf_vector, u_mlp, u_mf, \
                u_mlp_g, u_mf_g, \
                neg_mlp_vector, neg_mf_vector = self.model(
                    users, items, neg_items,
                    U_mlp_g=self.U_mlp_g[0], U_mf_g=self.U_mf_g[0])
            loss = self.pricdr_loss_fn(
                mlp_vector, mf_vector, u_mlp, u_mf, u_mlp_g, u_mf_g,
                neg_mlp_vector, neg_mf_vector)
            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "P2FCDR" in self.method:
            items, neg_items = interactions
            # mlp_vector, mf_vector: (batch_size, emb_size)
            # neg_mlp_vector, neg_mf_vector: (batch_size, num_neg, emb_size)
            mlp_vector, mf_vector, neg_mlp_vector, neg_mf_vector = \
                self.model(users, items, neg_items,
                           U_mlp_g=self.U_mlp_g[0], U_mf_g=self.U_mf_g[0])
            loss = self.p2fcdr_loss_fn(
                mlp_vector, mf_vector, neg_mlp_vector, neg_mf_vector)
            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "PPDM" in self.method:
            items, neg_items = interactions
            # u_mu, u_sigma, u_mu_g, u_sigma_g: (batch_size, emb_size)
            # v_mu, v_sigma: (batch_size, emb_size)
            # neg_v_mu, neg_v_sigma: (batch_size, num_neg, emb_size)
            u_mu, u_sigma, v_mu, v_sigma, \
                u_mu_g, u_sigma_g, \
                neg_v_mu, neg_v_sigma = self.model(
                    users, items, neg_items,
                    U_mu_g=self.U_mu_g[0], U_sigma_g=self.U_sigma_g[0])
            loss = self.ppdm_loss_fn(u_mu, u_sigma, v_mu, v_sigma,
                                     u_mu_g, u_sigma_g,
                                     neg_v_mu, neg_v_sigma,
                                     users, zeta, tilde_u_mu, tilde_u_sigma)
            if ("Fed" in self.method) and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        loss.backward()
        self.optimizer.step()
        self.step += 1
        return loss.item()

    def hi_loss_fn(self, u_e, u_g, v_e, neg_v_e,
                   neg_u_e, round):
        pos_score = self.discri(u_e, v_e)  # (batch_size, )
        neg_score = self.discri(u_e, neg_v_e)

        loss = -F.logsigmoid(pos_score).mean() \
            - F.logsigmoid(-neg_score).mean(dim=1).mean()

        if (u_g is not None) and round > self.args.wait_round:
            def mi_sp_and_glob(self, u_e, u_g, neg_u_e):
                pos_score = self.discri(u_e, u_g)
                neg_score = self.discri(neg_u_e, u_g)

                # pos_label, neg_label \
                #     = torch.ones(pos_score.size()).to(self.device), \
                #     torch.zeros(neg_score.size()).to(self.device)
                # kt_mi_loss = self.bce_criterion(pos_score, pos_label) \
                #     + self.bce_criterion(neg_score, neg_label)

                # kt_mi_loss = self.jsd_criterion(pos_score, neg_score)
                kt_mi_loss = self.hinge_criterion(pos_score, neg_score)

                kt_mi_loss = kt_mi_loss.mean()

                return kt_mi_loss
            kt_loss = mi_sp_and_glob(self, u_e, u_g.detach(), neg_u_e)
            loss += self.args.lam * kt_loss
        return loss

    def lo_loss_fn(self, u_s, aug_u_s, u_e, v_s, neg_v_s,
                   neg_u_s, z_s, aug_z_s, round):
        pos_score = self.discri(u_s, v_s)  # (batch_size, )
        neg_score = self.discri(u_s, neg_v_s)

        rating_loss = -F.logsigmoid(pos_score).mean() \
            - F.logsigmoid(-neg_score).mean(dim=1).mean()

        if round > self.args.wait_round:
            def mi_shr_and_sp(self, u_s, u_e, neg_u_s):
                pos_score = self.discri(u_s, u_e)
                neg_score = self.discri(neg_u_s, u_e)

                # pos_label, neg_label \
                #     = torch.ones(pos_score.size()).to(self.device), \
                #     torch.zeros(neg_score.size()).to(self.device)
                # kt_mi_loss = self.bce_criterion(pos_score, pos_label) \
                #     + self.bce_criterion(neg_score, neg_label)

                # kt_mi_loss = self.jsd_criterion(pos_score, neg_score)
                kt_mi_loss = self.hinge_criterion(pos_score, neg_score)

                kt_mi_loss = kt_mi_loss.mean()

                return kt_mi_loss
            kt_loss = mi_shr_and_sp(self, u_s, u_e.detach(), neg_u_s)
        else:
            kt_loss = 0

        def mi_node_and_graph(self, z_s, u_s, aug_u_s):
            pos_score = self.discri(z_s, u_s)  # (batch_size, )
            neg_score = self.discri(z_s, aug_u_s)

            pos_label, neg_label = torch.ones(pos_score.size())\
                .to(self.device), torch.zeros(neg_score.size()).to(self.device)
            contrastive_loss = self.bce_criterion(pos_score, pos_label) \
                + self.bce_criterion(neg_score, neg_label)
            contrastive_loss = contrastive_loss.mean()

            return contrastive_loss

        def mi_graph_and_node(self, z_s, u_s, aug_z_s):
            pos_score = self.discri(z_s, u_s)  # (batch_size, )
            neg_score = self.discri(aug_z_s, u_s)

            pos_label, neg_label = torch.ones(pos_score.size())\
                .to(self.device), torch.zeros(neg_score.size()).to(self.device)
            contrastive_loss = self.bce_criterion(pos_score, pos_label) \
                + self.bce_criterion(neg_score, neg_label)
            contrastive_loss = contrastive_loss.mean()

            return contrastive_loss

        contrastive_loss = mi_node_and_graph(self, z_s, u_s, aug_u_s) \
            + mi_graph_and_node(self, z_s, u_s, aug_z_s)

        loss = rating_loss + self.args.lam * kt_loss + self.args.gamma \
            * contrastive_loss
        return loss

    def hf_loss_fn(self, u, v, neg_v):
        pos_score = self.discri(u, v)  # (batch_size, )
        neg_score = self.discri(u, neg_v)

        loss = -F.logsigmoid(pos_score).mean() \
            - F.logsigmoid(-neg_score).mean(dim=1).mean()
        return loss

    def dhcf_loss_fn(self, u, v, neg_v):
        pos_score = self.discri(u, v)  # (batch_size, )
        neg_score = self.discri(u, neg_v)

        loss = -F.logsigmoid(pos_score).mean() \
            - F.logsigmoid(-neg_score).mean(dim=1).mean()
        return loss

    def mf_loss_fn(self, mlp_vector, mf_vector, neg_mlp_vector, neg_mf_vector):
        pos_score = self.discri(mlp_vector, mf_vector)  # (batch_size, )
        neg_score = self.discri(neg_mlp_vector, neg_mf_vector)

        loss = -F.logsigmoid(pos_score).mean() \
            - F.logsigmoid(-neg_score).mean(dim=1).mean()
        return loss

    def gnn_loss_fn(self, u, v, neg_v):
        pos_score = self.discri(u, v)  # (batch_size, )
        neg_score = self.discri(u, neg_v)

        loss = -F.logsigmoid(pos_score).mean() \
            - F.logsigmoid(-neg_score).mean(dim=1).mean()
        return loss

    def pricdr_loss_fn(self, mlp_vector, mf_vector, u_mlp, u_mf,
                       u_mlp_g, u_mf_g,
                       neg_mlp_vector, neg_mf_vector):
        pos_score = self.discri(mlp_vector, mf_vector)  # (batch_size, )
        neg_score = self.discri(neg_mlp_vector, neg_mf_vector)

        loss = -F.logsigmoid(pos_score).mean() \
            - F.logsigmoid(-neg_score).mean(dim=1).mean() \

        if (u_mlp_g is not None) and (u_mf_g is not None):
            lam_p = 1.0  # 1.0 for FKCB, SGHT, and 2.0 for SCEC is the best
            aligned_loss = torch.mean(
                torch.sum((u_mlp - u_mlp_g.detach())**2, axis=1)) \
                + torch.mean(
                torch.sum((u_mf - u_mf_g.detach())**2, axis=1))
            loss += lam_p * aligned_loss
        return loss

    def p2fcdr_loss_fn(self, mlp_vector, mf_vector,
                       neg_mlp_vector, neg_mf_vector):
        pos_score = self.discri(mlp_vector, mf_vector)  # (batch_size, )
        neg_score = self.discri(neg_mlp_vector, neg_mf_vector)

        loss = -F.logsigmoid(pos_score).mean() \
            - F.logsigmoid(-neg_score).mean(dim=1).mean() \

        return loss

    def ppdm_loss_fn(self, u_mu, u_sigma, v_mu, v_sigma,
                     u_mu_g, u_sigma_g,
                     neg_v_mu, neg_v_sigma,
                     users, zeta, tilde_u_mu, tilde_u_sigma):
        pos_score = self.discri(u_mu, v_mu) + \
            self.discri(u_sigma, v_sigma)  # (batch_size, )
        neg_score = self.discri(u_mu, neg_v_mu) + \
            self.discri(u_sigma, neg_v_sigma)

        loss = -F.logsigmoid(pos_score).mean() \
            - F.logsigmoid(-neg_score).mean(dim=1).mean()

        if (u_mu_g is not None) and (u_sigma_g is not None):
            # 0.1 for FKCB and SGHT is the best, 1.0 for SCEC is the best
            lam_p = 0.1
            aligned_loss = torch.mean(
                torch.sum((u_mu - u_mu_g.detach())**2, axis=1)) \
                + torch.mean(
                torch.sum((u_sigma - u_sigma_g.detach())**2, axis=1))
            loss += lam_p * aligned_loss

        if (zeta is not None) and (tilde_u_mu is not None) \
                and (tilde_u_sigma is not None):
            lam_c = 0  # 0 is the best
            co_clustering_loss = torch.mean(
                torch.sum(
                    (u_mu - tilde_u_mu[zeta[users]].detach())**2, axis=1)) \
                + torch.mean(
                    torch.sum(
                        (u_sigma - tilde_u_sigma[zeta[users]].detach())**2,
                        axis=1))
            loss += lam_c * co_clustering_loss

        return loss

    @ staticmethod
    def flatten(source):
        return torch.cat([value.flatten() for value in source])

    def prox_reg(self, params1, params2, mu):
        params1_values, params2_values = [], []
        # Record the model parameter aggregation results of each branch
        # separately
        for branch_params1, branch_params2 in zip(params1, params2):
            branch_params2 = [branch_params2[key]
                              for key in branch_params1.keys()]
            params1_values.extend(branch_params1.values())
            params2_values.extend(branch_params2)

        # Multidimensional parameters should be compressed into one dimension
        # using the flatten function
        s1 = self.flatten(params1_values)
        s2 = self.flatten(params2_values)
        return mu/2 * torch.norm(s1 - s2)

    def test_batch(self, users, interactions):
        """Tests the model for one batch.

        Args:
            users: Input user IDs.
            interactions: Input user interactions.
        """
        users = torch.LongTensor(users).to(self.device)
        interactions = [torch.LongTensor(x).to(
            self.device) for x in interactions]

        # items: (batch_size, )
        # neg_items: (batch_size, num_test_neg)
        items, neg_items = interactions
        # all_items: (batch_size, num_test_neg + 1)
        # Note that the elements in the first column are the positive samples.
        all_items = torch.hstack([items.reshape(-1, 1), neg_items])

        if self.method == "FedHCDR":
            # u_e: (batch_size, emb_size)
            # v_e: (batch_size, num_test_neg + 1, emb_size)
            u_e, v_e = self.model.forward_hi(users, all_items)
            u_s, v_s = self.model.forward_lo(users, all_items)

            # Use domain-exclusive + domain-shared user representations
            # together
            u = u_e + u_s
            v = v_e + v_s

            # Use domain-exclusive user representations only
            # u = u_e
            # v = v_e

            # Use domain-shared user representations only
            # u = u_s
            # v = v_s
        elif "HF" in self.method:
            u, v = self.model(users, all_items)
        elif "DHCF" in self.method:
            u, v = self.model(users, all_items)
        elif "PPDM" in self.method:
            u_mu, u_sigma, v_mu, v_sigma = self.model(users, all_items)
        elif ("MF" in self.method) or ("PriCDR" in self.method) or \
                ("P2FCDR" in self.method):
            mlp_vector, mf_vector = self.model(users, all_items)
        else:
            u, v = self.model(users, all_items)

        if "PPDM" in self.method:
            # result: (batch_size, num_test_neg + 1)
            result = self.discri(u_mu, v_mu) + self.discri(u_sigma, v_sigma)
        elif ("MF" in self.method) or ("PriCDR" in self.method) or \
                ("P2FCDR" in self.method):
            result = self.discri(mlp_vector, mf_vector)
        else:
            result = self.discri(u, v)

        # (batch_size, num_test_neg + 1)
        result = result.view(result.size()[0],
                             result.size()[1])

        pred = []
        for score in result:
            # score:  (num_test_neg + 1)
            # Note that the first one is the positive sample.
            # `(-score).argsort().argsort()` indicates where the elements at
            # each position are ranked in the list of logits in descending
            # order (since `argsort()` defaults to ascending order, we use
            # `-score` here). Since the first one is the positive sample,
            # then `...[0].item()` indicates the ranking of the positive
            # sample.
            rank = (-score).argsort().argsort()[0].item()
            pred.append(rank + 1)  # `+1` makes the ranking start from 1

        return pred
