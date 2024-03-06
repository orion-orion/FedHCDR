# -*- coding: utf-8 -*-
import numpy as np
import torch
from dataset import RecDataset
from local_graph import LocalGraph


def load_ratings_dataset(args):
    client_train_datasets = []
    client_valid_datasets = []
    client_test_datasets = []
    for domain in args.domains:
        if args.method == "FedHCDR":
            model = "HL_HF"
        else:
            model = args.method.replace("Fed", "")
            model = model.replace("Local", "")

        train_dataset = RecDataset(
            args, domain, model, mode="train", load_prep=args.load_prep)
        valid_dataset = RecDataset(
            args, domain, model, mode="valid", load_prep=args.load_prep)
        test_dataset = RecDataset(
            args, domain, model, mode="test", load_prep=args.load_prep)

        client_train_datasets.append(train_dataset)
        client_valid_datasets.append(valid_dataset)
        client_test_datasets.append(test_dataset)
    return client_train_datasets, client_valid_datasets, client_test_datasets


def load_graph_dataset(args, client_train_datasets):
    assert (args.method == "FedHCDR" or ("HF" in args.method)
            or ("DHCF" in args.method) or ("GNN" in args.method)
            or ("PPDM" in args.method))
    if args.method == "FedHCDR":
        UU_adjs = []
        VV_adjs = []
        Ms = []
        perturb_UU_adjs = []
    elif ("HF" in args.method) or ("DHCF" in args.method):
        UU_adjs = []
        VV_adjs = []
    else:
        all_adjs = []
    for train_dataset, domain in zip(client_train_datasets, args.domains):
        if args.method == "FedHCDR":
            model = "HL_HF"
        else:
            model = args.method.replace("Fed", "")
            model = model.replace("Local", "")
        local_graph = LocalGraph(args, domain,
                                 train_dataset.num_users,
                                 train_dataset.num_items,
                                 model, load_prep=args.load_prep)
        if args.method == "FedHCDR":
            UU_adjs.append(local_graph.UU_adj)
            VV_adjs.append(local_graph.VV_adj)
            Ms.append(local_graph.M)
            perturb_UU_adjs.append(local_graph.perturb_UU_adj)
        elif ("HF" in args.method) or ("DHCF" in args.method):
            UU_adjs.append(local_graph.UU_adj)
            VV_adjs.append(local_graph.VV_adj)
        else:
            all_adjs.append(local_graph.all_adj)

    if args.cuda:
        torch.cuda.empty_cache()
        device = "cuda:%s" % args.gpu
    else:
        device = "cpu"
    if args.method == "FedHCDR":
        for idx, (UU_adj, VV_adj, M, perturb_UU_adj) \
                in enumerate(zip(UU_adjs, VV_adjs, Ms, perturb_UU_adjs)):
            UU_adjs[idx] = UU_adj.to(device)
            VV_adjs[idx] = VV_adj.to(device)
            Ms[idx] = M.to(device)
            perturb_UU_adjs[idx] = perturb_UU_adj.to(device)
    elif ("HF" in args.method) or ("DHCF" in args.method):
        for idx, (UU_adj, VV_adj) in enumerate(zip(UU_adjs, VV_adjs)):
            UU_adjs[idx] = UU_adj.to(device)
            VV_adjs[idx] = VV_adj.to(device)
    else:
        for idx, all_adj in enumerate(all_adjs):
            all_adjs[idx] = all_adj.to(device)

    if args.method == "FedHCDR":
        return UU_adjs, VV_adjs, Ms, perturb_UU_adjs
    elif ("HF" in args.method) or ("DHCF" in args.method):
        return UU_adjs, VV_adjs
    elif ("GNN" in args.method) or ("PPDM" in args.method):
        return all_adjs


def init_clients_weight(clients):
    """Initialize the aggretation weight, which is the ratio of the number of
    samples per client to the total number of samples.
    """
    client_n_samples_train = [client.n_samples_train for client in clients]

    samples_sum_train = np.sum(client_n_samples_train)
    for client in clients:
        client.train_weight = client.n_samples_train / samples_sum_train
        # Here we need to average the model and representation by the same
        # weight for the validation / test
        client.valid_weight = 1 / len(clients)
        client.test_weight = 1 / len(clients)
