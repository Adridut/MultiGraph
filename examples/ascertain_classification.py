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
from dhg.models import HGNN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

def HGNNTrain():
    # set_seed(2021)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print_log("loading data")
    selected_modalities=['ECG']
    X, y, train_mask, test_mask, val_mask, _, _ = load_ASERTAIN(selected_modalities=selected_modalities, label = 'valence', train_ratio=60, val_ratio=20, test_ratio=20)

    print_log("generating hypergraph")
    X = torch.tensor(X)
    y = torch.tensor(y, dtype=torch.long)
    G = Hypergraph.from_feature_kNN(X, k=3)
    X, lbl = torch.eye(G.num_v), y
    train_mask = torch.tensor(train_mask)
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)
    n_classes = 2

    # X, lbl = torch.eye(data["num_vertices"]), data["labels"]
    # G = Hypergraph(data["num_vertices"], data["edge_list"])
    # train_mask = data["train_mask"]
    # val_mask = data["val_mask"]
    # test_mask = data["test_mask"]
    # n_classes = data["num_classes"]


    net = HGNN(X.shape[1], 32, n_classes, use_bn=False)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(net, X, G, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, G, lbl, val_mask)
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
    res = infer(net, X, G, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)

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
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "weighted"}}])
    net.eval()
    outs = net(X, A)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res

def singleHG():
    print_log("loading data")
    selected_modalities=['ECG']
    X_train, X_test, y_train, y_test, _, _ = load_ASERTAIN(selected_modalities=selected_modalities, label = 'arousal', train_ratio=80)

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, -1 * np.ones_like(y_test)])
    print(y)

    print_log("generating hypergraph")
    hg = gen_knn_hg(X, n_neighbors=25)

    print_log("learning on hypergraph")
    y_predict = trans_infer(hg, y, lbd=100)
    print_log("accuracy: {}".format(accuracy_score(y_test, y_predict)))
    print_log("f1: {}".format(f1_score(y_test, y_predict), average='weighted'))


def main():
    print_log("loading data")
    # selected_modalities=['ECG']
    selected_modalities=[['EEG'], ['GSR'], ['ECG'], ['EMO'], ['EEG', 'GSR'], ['EEG', 'ECG'], ['EEG', 'EMO']
                         , ['GSR', 'ECG'], ['GSR', 'EMO'], ['ECG', 'EMO'], ['EEG', 'GSR', 'ECG']
                         , ['EEG', 'GSR', 'EMO'], ['EEG', 'ECG', 'EMO'], ['GSR', 'ECG', 'EMO'], ['EEG', 'GSR', 'ECG', 'EMO']
                        ]
    all_X_train = []
    all_X_test = []
    for m in selected_modalities:
        X_train, X_test, y_train, y_test, subject_attributes, video_attributes = load_ASERTAIN(selected_modalities=m, label = 'valence', train_ratio=80)
        all_X_train.append(X_train)
        all_X_test.append(X_test)
    

    X = [np.vstack((all_X_train[imod], all_X_test[imod])) for imod in range(len(selected_modalities))]
    y = np.concatenate((y_train, -1 * np.ones_like(y_test)))

    print_log("generating hypergraph")
    # Modality HG
    hg_list = [
        gen_knn_hg(X[imod], n_neighbors=25)
        for imod in range(len(selected_modalities))
    ]

    # Attribute HG
    subject_attr_hg_list = [
        gen_attribute_hg(X[0].shape[0], a)
        for a in subject_attributes
    ]
    video_attr_hg_list = [
        gen_attribute_hg(X[0].shape[0], a)
        for a in video_attributes
    ]
    attr_hg = concat_multi_hg(subject_attr_hg_list + video_attr_hg_list)

    # Mod + Attr HG
    hg_list = [concat_multi_hg([hg, attr_hg]) for hg in hg_list]

    print_log("learning on hypergraph")
    y_predict = multi_hg_weighting_trans_infer(hg_list, y, lbd=100, max_iter=10, mu=0.00000001)
    print_log("accuracy: {}".format(accuracy_score(y_test, y_predict)))
    print_log("f1: {}".format(f1_score(y_test, y_predict), average='weighted'))


def is_column_feature(columns, column_index):
    print(columns, column_index)
    return ('label' not in columns[column_index] and 'id' not in columns[column_index])


if __name__ == "__main__":
    HGNNTrain()