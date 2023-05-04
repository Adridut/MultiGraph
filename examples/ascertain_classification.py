# coding=utf-8
import numpy as np

from data_helper import load_ASERTAIN

import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Hypergraph
from dhg.models import HGNN, HGNNP
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

from dhg.experiments import HypergraphVertexClassificationTask as Task
import torch.nn as nn
import torch.optim as optim

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, f1_score

from dhgnn import DHGNN
import matplotlib.pyplot as plt

from fc import FC

def print_log(message):
    """
    :param message: str,
    :return:
    """
    print("[{}] {}".format(time.strftime("%Y-%m-%d %X", time.localtime()), message))

def run_baseline(selected_modalities, label, train_ratio, val_ratio, test_ratio, model, trials):
    acc = 0
    f1 = 0
    if model == "SVM":
        model = svm.SVC()
    elif model == "NB":
        model = GaussianNB()
    for i in range(trials):
        X, y, _, _, _, _, _, _, _ = load_ASERTAIN(selected_modalities[0], label, train_ratio, val_ratio, test_ratio)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        y_pred = model.fit(X_train, y_train).predict(X_test)
        evaluator = Evaluator(["accuracy", "f1_score"])
        y_pred = torch.from_numpy(y_pred)
        y_test = torch.from_numpy(y_test)
        res = evaluator.test(y_test, y_pred)
        acc += (res['accuracy'])
        f1 += (res['f1_score'])
    
    print(acc/trials, f1/trials)


def run(device, X, lbl, train_mask, test_mask, val_mask, G, net, lr , weight_decay, n_epoch, model_name):

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(n_epoch):
        # train
        train(net, X, G, lbl, train_mask, optimizer, epoch, model_name)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res, _ = infer(net, X, G, lbl, val_mask, model_name)
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
    res, all_outs = infer(net, X, G, lbl, test_mask, model_name, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
    return res, all_outs


def train(net, X, A, lbls, train_idx, optimizer, epoch, model_name):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    if model_name == "FC":
        outs = net(X)   
    else:
        outs = net(X, A)

    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, A, lbls, idx, model_name, test=False):
    evaluator = Evaluator(["accuracy", "f1_score"])
    net.eval()
    if model_name == "FC":
        all_outs = net(X)
    else:
        all_outs = net(X, A)

    outs, lbls = all_outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res, all_outs


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

if __name__ == "__main__":
    # set_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    selected_modalities = [['ECG'], ['EEG'], ['GSR'], ['EMO']]
    # selected_modalities = [['EMO']]
    # selected_modalities = [['ECG', 'EEG', 'GSR']]
    # selected_modalities = [['ECG', 'EEG', 'EMO', 'GSR']]
    # selected_modalities=[['ECG'], ['EEG'], ['EMO'], ['GSR'], ['ECG', 'EEG'], ['ECG', 'EMO'], ['ECG', 'GSR'], ['EEG', 'EMO'], ['EEG', 'GSR'], ['EMO', 'GSR'], ['ECG', 'EEG', 'EMO'], ['ECG', 'EEG', 'GSR'], ['ECG', 'EMO', 'GSR'], ['EEG', 'EMO', 'GSR'], ['ECG', 'EEG', 'EMO', 'GSR']]

    label = "arousal"
    train_ratio = 80
    val_ratio = 10
    test_ratio = 10
    n_classes = 2
    n_hidden_layers = 8
    k = 4 #4, 20
    lr = 0.001 #0.01, 0.001
    weight_decay = 5*10**-4
    n_epoch = 1
    model_name = "HGNNP" #HGNN, HGNNP, NB, SVM
    n_nodes = 2088
    fuse_models = True
    use_attributes = True


    final_acc = 0
    final_f1 = 0
    trials = 1
    all_accs = [0 for m in selected_modalities]
    all_f1s = [0 for m in selected_modalities]


    if model_name == "NB" or model_name == "SVM":
        run_baseline(selected_modalities, label, train_ratio, val_ratio, test_ratio, model_name, trials)

    else:
        model = select_model(feat_dimension=n_nodes, n_hidden_layers=n_hidden_layers, n_classes=n_classes, model=model_name)
        for trial in range(trials):
            print_log("trial: " + str(trial))
            i = 0
            inputs = []
            for m in selected_modalities:

                print_log("loading data: " + str(m))
                X, y, train_mask, test_mask, val_mask, sa, va, lpa, hpa = load_ASERTAIN(selected_modalities=m, label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
                
                X = torch.tensor(X).float()

                print_log("generating hypergraph: " + str(m))
                G = Hypergraph(X.size()[0])

                for mod in m:
                    x, y, train_mask, test_mask, val_mask, sa, va, lpa, hpa = load_ASERTAIN(selected_modalities=[mod], label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
                    x = torch.tensor(x).float()
                    G.add_hyperedges_from_feature_kNN(x, k=k, group_name=mod)

                if use_attributes:
                    for a in sa:
                        G.add_hyperedges(a, group_name="subject_attributes_"+str(a))
                    for a in va:
                        G.add_hyperedges(a, group_name="video_attributes_"+str(a))
                    for a in lpa:
                        G.add_hyperedges(a, group_name="low_personality_attributes_"+str(a))
                    for a in hpa:
                        G.add_hyperedges(a, group_name="high_personality_attributes_"+str(a))

                y = torch.from_numpy(y).long()
                train_mask = torch.tensor(train_mask)
                val_mask = torch.tensor(val_mask)
                test_mask = torch.tensor(test_mask)
                X = torch.eye(G.num_v)

                G.to(device)
                X = X.to(device)
                y = y.to(device)

                res, out = run(device, X, y, train_mask, test_mask, val_mask, G, model, lr , weight_decay, n_epoch, model_name)
                all_accs[i] += res['accuracy']
                all_f1s[i] += res['f1_score']
                inputs.append(out)
                i += 1

            if fuse_models:
                print_log("fusing models")
                inputs = torch.cat(inputs, 1)
                print(inputs.size())
                # inputs = torch.tensor(inputs).float()
                # inputs = inputs.permute(1,0)
                net = FC(inputs.size()[1], n_classes)
                final_res, _ = run(device, inputs, y, train_mask, test_mask, val_mask, G, net, lr , weight_decay, n_epoch, "FC")
                final_acc += final_res['accuracy']
                final_f1 += final_res['f1_score']


        print("acc: ", np.divide(all_accs,trials))
        print("f1: ", np.divide(all_f1s,trials))
        print(selected_modalities)

        if fuse_models:
            print("final acc: ", final_acc/trials)
            print("final f1: ", final_f1/trials)

    









    
