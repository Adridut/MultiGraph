# coding=utf-8
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from hyperg.generation import gen_knn_hg, gen_attribute_hg, concat_multi_hg, fuse_mutli_sub_hg, gen_clustering_hg
from hyperg.learning import trans_infer, multi_hg_trans_infer, multi_hg_weighting_trans_infer, tensor_hg_trans_infer
from hyperg.utils import print_log

from data_helper import load_ASERTAIN

from hyperg.learning import multi_hg_trans_infer

import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Hypergraph
from dhg.data import Cooking200
from dhg.models import HGNN, HGNNP
from multihgnn import MultiHGNN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

from numpy import array
from numpy import argmax
from numpy import tensordot

from FusionLayer import FusionLayer


# def multiHGNNTrain(device, Xs, y, train_mask, test_mask, val_mask):
#     print_log("generating hypergraph")
#     G = []
#     X = []
#     for x in Xs:
#         G.append(Hypergraph.from_feature_kNN(x, k=3))
#         X.append(x)
#         lbl = y

#     n_classes = 2

#     net = MultiHGNN(X, 8, n_classes, use_bn=True)
#     optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)

#     # X, lbl = X.to(device), lbl.to(device)
#     # G = G.to(device)
#     net = net.to(device)

#     best_state = None
#     best_epoch, best_val = 0, 0
#     for epoch in range(200):
#         # train
#         train(net, X, G, lbl, train_mask, optimizer, epoch)
#         # validation
#         if epoch % 1 == 0:
#             with torch.no_grad():
#                 val_res, _ = infer(net, X, G, lbl, val_mask)
#             if val_res > best_val:
#                 print(f"update best: {val_res:.5f}")
#                 best_epoch = epoch
#                 best_val = val_res
#                 best_state = deepcopy(net.state_dict())
#     print("\ntrain finished!")
#     print(f"best val: {best_val:.5f}")
#     # test
#     print("test...")
#     net.load_state_dict(best_state)
#     res, all_outs = infer(net, X, G, lbl, test_mask, test=True)
#     print(f"final result: epoch: {best_epoch}")
#     print(res)
#     return all_outs, res


# def fusion(X, y, train_mask, val_mask, test_mask, w):
#     # Concatenate the HGNN features along the feature dimension
#     # X = torch.cat(X, dim=1)

#     # Define the fusion net
#     fusion_net = FusionLayer(len(X), w)
#     optimizer = optim.Adam(fusion_net.parameters(), lr=0.0001, weight_decay=5e-4)


#     best_state = None
#     best_epoch, best_val = 0, 0
#     for epoch in range(200):
#         # train
#         train_fusion(fusion_net, X, y, train_mask, optimizer, epoch, w)
#         # validation
#         if epoch % 1 == 0:
#             with torch.no_grad():
#                 val_res, _ = infer_fusion(fusion_net, X, y, val_mask, w)
#             if val_res > best_val:
#                 print(f"update best: {val_res:.5f}")
#                 best_epoch = epoch
#                 best_val = val_res
#                 best_state = deepcopy(fusion_net.state_dict())
#     print("\ntrain finished!")
#     print(f"best val: {best_val:.5f}")
#     # test
#     print("test...")
#     fusion_net.load_state_dict(best_state)
#     res, w = infer_fusion(fusion_net, X, y, test_mask, w, test=True)
#     print(f"final result: epoch: {best_epoch}")
#     print(res)
#     print(w)

def run_GHGNN(device, selected_modalities, label, train_ratio, val_ratio, test_ratio, n_classes, n_hidden_layers):

    first_HG = True
    G = Hypergraph

    for m in selected_modalities:

        print_log("loading data: " + str(m))
        X, y, train_mask, test_mask, val_mask, sa, va = load_ASERTAIN(selected_modalities=m, label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        X = torch.tensor(X).float()
        y = torch.from_numpy(y).long()
        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)

        print_log("generating hypergraph: " + str(m))
        if first_HG:
            G = Hypergraph.from_feature_kNN(X, k=3)
            first_HG = False
        else:
            G.add_hyperedges_from_feature_kNN(X, k=3, group_name=str(m))

        X, lbl = torch.eye(G.num_v), y


    # Attribute HG
    for a in sa:
        G.add_hyperedges(a, group_name="subject_attributes")

    for a in va:
        G.add_hyperedges(a, group_name="video_attributes")

    net = HGNNP(X.shape[1], n_hidden_layers, n_classes, use_bn=True)
    outs, res = HGNNTrain(device, X, y, train_mask, test_mask, val_mask, G, net)


def run_HGNN(device, selected_modalities, label, train_ratio, val_ratio, test_ratio, n_classes, n_hidden_layers):

    hgnn_acc = []
    for m in selected_modalities:

        print_log("loading data: " + str(m))
        X, y, train_mask, test_mask, val_mask, sa, va = load_ASERTAIN(selected_modalities=m, label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        X = torch.tensor(X).float()
        y = torch.from_numpy(y).long()
        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)

        print_log("generating hypergraph: " + str(m))
        G = Hypergraph.from_feature_kNN(X, k=3)
        X, lbl = torch.eye(G.num_v), y
        # Attribute HG
        for a in sa:
            G.add_hyperedges(a)

        for a in va:
            G.add_hyperedges(a)

        net = HGNN(X.shape[1], n_hidden_layers, n_classes, use_bn=True)
        outs, res = HGNNTrain(device, X, y, train_mask, test_mask, val_mask, G, net)
        hgnn_acc.append(res["accuracy"])


    print(hgnn_acc)


def HGNNTrain(device, X, lbl, train_mask, test_mask, val_mask, G, net):

    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(2000):
        # train
        train(net, X, G, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res, _ = infer(net, X, G, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res, all_outs = infer(net, X, G, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
    return all_outs, res

# def train_fusion(net, X, lbls, train_idx, optimizer, epoch, w):
#     net.train()

#     st = time.time()
#     optimizer.zero_grad()
#     outs, _ = net(X, w)
#     outs, lbls = outs[train_idx], lbls[train_idx]
#     loss = F.cross_entropy(outs, lbls)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
#     return loss.item()


# @torch.no_grad()
# def infer_fusion(net, X, lbls, idx, w, test=False):
#     evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "macro"}}])
#     net.eval()
#     all_outs, w = net(X, w)
#     outs, lbls = all_outs[idx], lbls[idx]
#     if not test:
#         res = evaluator.validate(lbls, outs)
#     else:
#         res = evaluator.test(lbls, outs)
#     return res, w

def train(net, X, A, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, A)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, A, lbls, idx, test=False):
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "macro"}}])
    net.eval()
    all_outs = net(X, A)
    outs, lbls = all_outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res, all_outs


if __name__ == "__main__":
    # set_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    selected_modalities=[['ECG'], ['EEG'], ['EMO'], ['GSR'], ['ECG', 'EEG'], ['ECG', 'EMO'], ['ECG', 'GSR'], ['EEG', 'EMO'], ['EEG', 'GSR'], ['EMO', 'GSR'], ['ECG', 'EEG', 'EMO'], ['ECG', 'EEG', 'GSR'], ['ECG', 'EMO', 'GSR'], ['EEG', 'EMO', 'GSR'], ['ECG', 'EEG', 'EMO', 'GSR']]
    # selected_modalities=[['ECG'], ['EEG'], ['EMO'], ['GSR']]
    # selected_modalities=[['ECG'], ['EEG']]
    label = "arousal"
    train_ratio = 80
    val_ratio = 10
    test_ratio = 10
    n_classes = 2
    n_hidden_layers = 8

    run_GHGNN(device, selected_modalities, label, train_ratio, val_ratio, test_ratio, n_classes, n_hidden_layers)






    
