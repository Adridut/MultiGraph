# coding=utf-8
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from hyperg.generation import gen_knn_hg, gen_attribute_hg, concat_multi_hg, fuse_mutli_sub_hg, gen_clustering_hg
from hyperg.learning import trans_infer, multi_hg_trans_infer, multi_hg_weighting_trans_infer, tensor_hg_trans_infer
from hyperg.utils import print_log

from data_helper import load_ASERTAIN


import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Hypergraph
from dhg.data import Cooking200
from dhg.models import HGNN, HGNNP, UniSAGE, UniGAT, HyperGCN, HNHN
from multihgnn import MultiHGNN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from dhg.utils import remap_edge_lists, edge_list_to_adj_list

from numpy import array
from numpy import argmax
from numpy import tensordot

from FusionLayer import FusionLayer

from dhg.experiments import HypergraphVertexClassificationTask as Task
import torch.nn as nn
import torch.optim as optim

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, f1_score
import statistics

from dhgnn import DHGNN
import matplotlib.pyplot as plt


def run_baseline(selected_modalities, label, train_ratio, val_ratio, test_ratio, model, trials):
    acc = 0
    f1 = 0
    if model == "SVM":
        model = svm.SVC()
    elif model == "NB":
        model = GaussianNB()
    for i in range(trials):
        X, y, _, _, _, _, _, _, _ = load_ASERTAIN(selected_modalities[0], label, train_ratio, val_ratio, test_ratio)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        y_pred = model.fit(X_train, y_train).predict(X_test)
        evaluator = Evaluator(["accuracy", "f1_score"])
        y_pred = torch.from_numpy(y_pred)
        y_test = torch.from_numpy(y_test)
        res = evaluator.test(y_test, y_pred)
        acc += (res['accuracy'])
        f1 += (res['f1_score'])
    
    print(acc/trials, f1/trials)


def run_HGNN(device, selected_modalities, label, train_ratio, val_ratio, test_ratio, n_classes, n_hidden_layers, k, lr , weight_decay, epoch, model):

    hgnn_acc = []
    hgnn_f1 = []
    outs = []
    for m in selected_modalities:

        print_log("loading data: " + str(m))
        X, y, train_mask, test_mask, val_mask, sa, va, lpa, hpa = load_ASERTAIN(selected_modalities=m, label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        X = torch.tensor(X).float()
        y = torch.from_numpy(y).long()
        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)

        print_log("generating hypergraph: " + str(m))
        G = Hypergraph.from_feature_kNN(X, k=k)
        X = torch.eye(G.num_v)

        G.to(device)
        X = X.to(device)
        # Attribute HG
        for a in sa:
            G.add_hyperedges(a, group_name="subject_attributes")


        for a in va:
            G.add_hyperedges(a, group_name="video_attributes")


        for a in lpa:
            G.add_hyperedges(a, group_name="low_personality_attributes")


        for a in hpa:
            G.add_hyperedges(a, group_name="high_personality_attributes")


        if model == "DHGNN":
            # net = DHGNN(dim_feat=X.shape[1],
            # n_categories=n_classes,
            # k_structured=128,
            # k_nearest=64,
            # k_cluster=64,
            # wu_knn=0,
            # wu_kmeans=10,
            # wu_struct=5,
            # clusters=400,
            # adjacent_centers=1,
            # n_layers=2,
            # layer_spec=[256],
            # dropout_rate=0.5,
            # has_bias=True
            # )
            net = DHGNN(dim_feat=X.shape[1],
            n_categories=n_classes,
            k_structured=128,
            k_nearest=64,
            k_cluster=64,
            wu_knn=0,
            wu_kmeans=10,
            wu_struct=5,
            clusters=400,
            adjacent_centers=5,
            n_layers=2,
            layer_spec=[256],
            dropout_rate=0.5,
            has_bias=True
            )
        else:
            net = HGNN(X.shape[1], n_hidden_layers, n_classes, use_bn=True)

        res, out = HGNNTrain(device, X, y, train_mask, test_mask, val_mask, G, net, lr , weight_decay, epoch, model)
        hgnn_acc.append(res["accuracy"])
        hgnn_f1.append(res["f1_score"])
        outs.append(out)


    return outs, y, hgnn_acc, hgnn_f1, test_mask


def run(device, X, lbl, train_mask, test_mask, val_mask, G, net, lr , weight_decay, epoch, model):

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(epoch):
        # train
        train(net, X, G, lbl, train_mask, optimizer, epoch, model)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res, _ = infer(net, X, G, lbl, val_mask, model)
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
    res, all_outs = infer(net, X, G, lbl, test_mask, model, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
    return res, all_outs


def train(net, X, A, lbls, train_idx, optimizer, epoch, model):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    if model == "DHGNN":
        ids = [i for i in range(2088) if train_idx[i]]
        outs = net(ids=ids, feats=X, G=A, edge_dict=A.nbr_e, ite=epoch)
        lbls = lbls[train_idx]   
    else:
        outs = net(X, A)
        outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, A, lbls, idx, model, test=False):
    evaluator = Evaluator(["accuracy", "f1_score"])
    net.eval()
    if model == "DHGNN":
        ids = [i for i in range(2088) if idx[i]]
        outs = net(ids=ids, feats=X, G=A, edge_dict=A.nbr_e, ite=epoch)
        lbls = lbls[idx]
    else:
        outs = net(X, A)
        outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res, outs


def select_model(feat_dimension, n_hidden_layers, n_classes, model):
        if model == "HGNN":
            return HGNN(feat_dimension, n_hidden_layers, n_classes, use_bn=True)
        elif model == "HGNNP":
            return HGNNP(feat_dimension, n_hidden_layers, n_classes, use_bn=True)
        elif model == "DHGNN":
            return DHGNN(dim_feat=feat_dimension,
            n_categories=n_classes,
            k_structured=128,
            k_nearest=64,
            k_cluster=64,
            wu_knn=0,
            wu_kmeans=10,
            wu_struct=5,
            clusters=400,
            adjacent_centers=1,
            n_layers=2,
            layer_spec=[256],
            dropout_rate=0.5,
            has_bias=True
            )


def generate_hypergraph(X, k, sa, va, lpa, hpa, use_attributes = True):
    G = Hypergraph.from_feature_kNN(X, k=k)
    if use_attributes:
        for a in sa:
            G.add_hyperedges(a, group_name="subject_attributes")
        for a in va:
            G.add_hyperedges(a, group_name="video_attributes")
        for a in lpa:
            G.add_hyperedges(a, group_name="low_personality_attributes")
        for a in hpa:
            G.add_hyperedges(a, group_name="high_personality_attributes")
    return G

if __name__ == "__main__":
    # set_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    selected_modalities = [['ECG'], ['EEG'], ['GSR'], ['EMO']]
    # selected_modalities = [['GSR']]
    # selected_modalities = [['ECG', 'EEG', 'GSR']]
    # selected_modalities = [['ECG', 'EEG', 'EMO', 'GSR']]
    # selected_modalities=[['ECG'], ['EEG'], ['EMO'], ['GSR'], ['ECG', 'EEG'], ['ECG', 'EMO'], ['ECG', 'GSR'], ['EEG', 'EMO'], ['EEG', 'GSR'], ['EMO', 'GSR'], ['ECG', 'EEG', 'EMO'], ['ECG', 'EEG', 'GSR'], ['ECG', 'EMO', 'GSR'], ['EEG', 'EMO', 'GSR'], ['ECG', 'EEG', 'EMO', 'GSR']]

    label = "valence"
    train_ratio = 90
    val_ratio = 5
    test_ratio = 5
    n_classes = 2
    n_hidden_layers = 8
    k = 4 #4, 20
    lr = 0.001 #0.01, 0.001
    weight_decay = 5*10**-4
    epoch = 20
    model = "HGNN"
    n_nodes = 2088


    accs = 0
    f1s = 0
    trials = 3
    all_accs = [0 for m in selected_modalities]
    all_f1s = [0 for m in selected_modalities]


    # run_baseline(selected_modalities, label, train_ratio, val_ratio, test_ratio, "NB", trials)


    model = select_model(feat_dimension=n_nodes, n_hidden_layers=n_hidden_layers, n_classes=n_classes, model=model)
    for trial in range(trials):
        print_log("trial: " + str(trial))
        i = 0
        for m in selected_modalities:

            print_log("loading data: " + str(m))
            X, y, train_mask, test_mask, val_mask, sa, va, lpa, hpa = load_ASERTAIN(selected_modalities=m, label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
            
            X = torch.tensor(X).float()
            y = torch.from_numpy(y).long()
            train_mask = torch.tensor(train_mask)
            val_mask = torch.tensor(val_mask)
            test_mask = torch.tensor(test_mask)

            print_log("generating hypergraph: " + str(m))
            G = generate_hypergraph(X, k, sa, va, lpa, hpa, use_attributes=True)
            X = torch.eye(G.num_v)

            G.to(device)
            X = X.to(device)
            y = y.to(device)

            res, out = run(device, X, y, train_mask, test_mask, val_mask, G, model, lr , weight_decay, epoch, model)
            all_accs[i] += res['accuracy']
            all_f1s[i] += res['f1_score']
            i += 1

    print("acc: ", np.divide(all_accs,trials))
    print("f1: ", np.divide(all_f1s,trials))

    









    
