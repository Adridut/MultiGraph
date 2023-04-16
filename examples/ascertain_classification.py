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




# def multiHGNNTrain(device, Xs, y, train_ratio, val_ratio, test_ratio, n_classes, n_hidden_layers, k, lr , weight_decay):
#     hgnn_acc = []
#     hgnn_f1 = []
#     outs = []
#     for m in selected_modalities:

#         print_log("loading data: " + str(m))
#         X, y, train_mask, test_mask, val_mask, sa, va = load_ASERTAIN(selected_modalities=m, label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
#         X = torch.tensor(X).float()
#         y = torch.from_numpy(y).long()
#         train_mask = torch.tensor(train_mask)
#         val_mask = torch.tensor(val_mask)
#         test_mask = torch.tensor(test_mask)

#         print_log("generating hypergraph: " + str(m))
#         G = Hypergraph.from_feature_kNN(X, k=k)
#         X = torch.eye(G.num_v)

#     net = MultiHGNN(X.shape[1], n_hidden_layers, n_classes, use_bn=True)
#     optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)


#     net = net.to(device)

#     best_state = None
#     best_epoch, best_val = 0, 0
#     for epoch in range(200):
#         # train
#         train(net, X, G, y, train_mask, optimizer, epoch)
#         # validation
#         if epoch % 1 == 0:
#             with torch.no_grad():
#                 val_res, _ = infer(net, X, G, y, val_mask)
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
#     res, all_outs = infer(net, X, G, y, test_mask, test=True)
#     print(f"final result: epoch: {best_epoch}")
#     print(res)
#     return res


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

def run_NB(selected_modalities, label, train_ratio, val_ratio, test_ratio):
    acc = 0
    f1 = 0
    for i in range(10):
        X, y, _, _, _, _, _, _, _ = load_ASERTAIN(selected_modalities[0], label, train_ratio, val_ratio, test_ratio)
        gnb = svm.SVC()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
        y_pred = gnb.fit(X_train, y_train).predict(X_test)
        evaluator = Evaluator(["accuracy", "f1_score"])
        y_pred = torch.from_numpy(y_pred)
        y_test = torch.from_numpy(y_test)
        res = evaluator.test(y_test, y_pred)
        acc += (res['accuracy'])
        f1 += (res['f1_score'])
    
    print(acc/10, f1/10)



def structure_builder(trial):
    i = 0
    first_HG = True
    k = trial.suggest_int("k", 1, 30)
    for X in Xs:
        print_log("generating hypergraph: " + str(selected_modalities[i]))
        if first_HG:
            # global G 
            G = Hypergraph.from_feature_kNN(X, k=k)
            first_HG = False
        else:
            G.add_hyperedges_from_feature_kNN(X, k=k, group_name="Modality_"+str(i))
            
        i += 1

    for a in sa:
        G.add_hyperedges(a, group_name="subject_attributes")

    for a in va:
        G.add_hyperedges(a, group_name="video_attributes")

    for a in lpa:
        G.add_hyperedges(a, group_name="low_personality_attributes")

    for a in hpa:
        G.add_hyperedges(a, group_name="high_personality_attributes")

    return G

def model_builder(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 1, 128)
    use_bn = True
    return HGNN(2088, hidden_dim, n_classes, use_bn=use_bn)


def train_builder(trial, model):
    optimizer = optim.Adam(
        model.parameters(),
        lr=trial.suggest_float("lr", 1e-4, 1e-2),
        weight_decay=trial.suggest_float("weight_decay", 1e-4, 1e-2),
    )
    criterion = nn.CrossEntropyLoss()
    return {
        "optimizer": optimizer,
        "criterion": criterion,
    }

def run_GHGNN(device, selected_modalities, label, train_ratio, val_ratio, test_ratio, n_classes, n_hidden_layers, k, lr , weight_decay, epoch):

    global va, sa, lpa, hpa, Xs
    Xs = []
    for m in selected_modalities:

        print_log("loading data: " + str(m))
        X, y, train_mask, test_mask, val_mask, sa, va, lpa, hpa = load_ASERTAIN(selected_modalities=m, label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        X = torch.tensor(X).float()
        y = torch.from_numpy(y).long()
        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)

        Xs.append(X)
        first_HG = True
        print_log("generating hypergraph: " + str(m))
        if first_HG:
            G = Hypergraph.from_feature_kNN(X, k=k)
            first_HG = False
        else:
            G.add_hyperedges_from_feature_kNN(X, k=k, group_name=str(m))

        X, lbl = torch.eye(2088), y


    # Attribute HG
    for a in sa:
        G.add_hyperedges(a, group_name="subject_attributes")

    for a in va:
        G.add_hyperedges(a, group_name="video_attributes")

    for a in lpa:
        G.add_hyperedges(a, group_name="low_personality_attributes")

    for a in hpa:
        G.add_hyperedges(a, group_name="high_personality_attributes")

    input_data = {
        "features": X,
        "labels": lbl,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }

    evaluator = Evaluator(["accuracy", "f1_score"])

    work_root = "D:\Dev\THU-HyperG\examples\logs"

    # task = Task(
    #     work_root, input_data, model_builder, train_builder, evaluator, device, structure_builder=structure_builder,
    # )

    # task.run(200, 50, "maximize")

    net = HGNN(X.shape[1], n_hidden_layers, n_classes, use_bn=True)
    res, out = HGNNTrain(device, X, y, train_mask, test_mask, val_mask, G, net, lr , weight_decay, epoch)
    acc = res["accuracy"]
    f1 = res["f1_score"]

    return acc, f1


def run_HGNN(device, selected_modalities, label, train_ratio, val_ratio, test_ratio, n_classes, n_hidden_layers, k, lr , weight_decay, epoch):

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
        net = HGNN(X.shape[1], n_hidden_layers, n_classes, use_bn=True)
        res, out = HGNNTrain(device, X, y, train_mask, test_mask, val_mask, G, net, lr , weight_decay, epoch)
        hgnn_acc.append(res["accuracy"])
        hgnn_f1.append(res["f1_score"])
        outs.append(out)


    return outs, y, hgnn_acc, hgnn_f1, test_mask


def HGNNTrain(device, X, lbl, train_mask, test_mask, val_mask, G, net, lr , weight_decay, epoch):

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(epoch):
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
    return res, all_outs


def train(net, X, A, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, A)
    # ids = [i for i in range(2088) if train_idx[i]]
    # outs = net(ids=ids, feats=X, G=A, edge_dict=A.nbr_e, ite=epoch)
    # lbls = lbls[train_idx]
    # print(A.nbr_e)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, A, lbls, idx, test=False):
    evaluator = Evaluator(["accuracy", "f1_score"])
    net.eval()
    outs = net(X, A)
    # ids = [i for i in range(2088) if idx[i]]
    # outs = net(ids=ids, feats=X, G=A, edge_dict=A.nbr_e, ite=epoch)
    # lbls = lbls[idx]
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res, outs


if __name__ == "__main__":
    # set_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    selected_modalities = [['ECG'], ['EEG'], ['GSR'], ['EMO']]
    # selected_modalities = [['ECG', 'EEG', 'EMO', 'GSR']]
    # selected_modalities=[['ECG'], ['EEG'], ['EMO'], ['GSR'], ['ECG', 'EEG'], ['ECG', 'EMO'], ['ECG', 'GSR'], ['EEG', 'EMO'], ['EEG', 'GSR'], ['EMO', 'GSR'], ['ECG', 'EEG', 'EMO'], ['ECG', 'EEG', 'GSR'], ['ECG', 'EMO', 'GSR'], ['EEG', 'EMO', 'GSR'], ['ECG', 'EEG', 'EMO', 'GSR']]

    label = "valence"
    train_ratio = 80
    val_ratio = 10
    test_ratio = 10
    n_classes = 2
    # n_hidden_layers = 32
    # k = 14
    # n_hidden_layers = 17
    # k = 27
    # n_hidden_layers = 85
    # k = 29
    n_hidden_layers = 8
    k = 4
    # lr = 0.001
    # weight_decay = 0.001
    # n_hidden_layers = 128
    # k = 64
    lr = 0.01
    weight_decay = 0.0005
    epoch = 200


    # run_NB(selected_modalities, label, train_ratio, val_ratio, test_ratio)
    # run_HGNN(device, selected_modalities, label, train_ratio, val_ratio, test_ratio, n_classes, n_hidden_layers)
    # run_GHGNN(device, selected_modalities, label, train_ratio, val_ratio, test_ratio, n_classes, n_hidden_layers, k, lr , weight_decay)
    
    
    accs = 0
    f1s = 0
    trials = 5
    all_accs = [0 for m in selected_modalities]
    all_f1s = [0 for m in selected_modalities]

    # for i in range(trials):
    #     print("trial: ", i)
    #     acc, f1 = run_GHGNN(device, selected_modalities, label, train_ratio, val_ratio, test_ratio, n_classes, n_hidden_layers, k, lr , weight_decay, epoch)
    #     accs += acc 
    #     f1s += f1

    # print("acc: ", accs/trials)
    # print("f1: ", f1s/trials)

    for i in range(trials):
        print("trial: ", i)
        out, y, all_acc, all_f1, test_mask = run_HGNN(device, selected_modalities, label, train_ratio, val_ratio, test_ratio, n_classes, n_hidden_layers, k, lr , weight_decay, epoch)
        out = torch.stack(out)

        for i in range(len(all_f1)):
            out[i] = torch.mul(out[i], (all_acc[i]*100 + all_f1[i]*100)*10)

        out = torch.mean(out, axis=0)
        out = F.softmax(out, dim=1)
        out = torch.argmax(out, axis=1)


        accs += accuracy_score(y[test_mask], out)
        f1s += f1_score(y[test_mask], out, average="macro")
        all_accs = [sum(x) for x in zip(all_accs, all_acc)]
        all_f1s = [sum(x) for x in zip(all_f1s, all_f1)]


    print("accuracy: ", accs/trials)
    print("f1: ", f1s/trials)
    print("all_accs: ", [x/trials for x in all_accs])
    print("all_f1s: ", [x/trials for x in all_f1s])


    # print(classification_report(y, out, target_names=["low", "high"]))




    
