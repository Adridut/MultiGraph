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

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, f1_score

from dhgnn import DHGNN
import matplotlib.pyplot as plt

from fc import FC
import random
from dhg.random import set_seed


# set_seed(0)

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
        print_log("trial: " + str(i))
        X, y, _, _, _, _, _, _, _ = load_ASERTAIN(selected_modalities[0], label, train_ratio, val_ratio, test_ratio, i)
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
        if epoch > best_epoch+200:
            break
        # train
        train(net, X, G, lbl, train_mask, optimizer, epoch, model_name, device)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res, _ = infer(net, X, G, lbl, val_mask, epoch, model_name, device)
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
    res, all_outs = infer(net, X, G, lbl, test_mask, best_epoch, model_name, device, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
    return res, all_outs


def train(net, X, A, lbls, train_idx, optimizer, epoch, model_name, device):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    
    if model_name == "FC":
        outs = net(X)   
        outs = outs[train_idx]
    elif model_name == "DHGNN":
        #ids: indices selected during train/valid/test, torch.LongTensor
        ids = [i for i in range(X.size()[0])]
        ids = torch.tensor(ids).long()[train_idx].to(device)
        outs = net(ids=ids, feats=X, edge_dict=A.e_list, G=A.H, ite=epoch, device=device)
    else:
        print(X.size())
        outs = net(X, A)
        outs = outs[train_idx]

    lbls = lbls[train_idx]
    loss = F.binary_cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    if epoch % 1 == 0:
        print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, A, lbls, idx, epoch, model_name, device, test=False):
    evaluator = Evaluator(["accuracy", "f1_score"])
    net.eval()
    if model_name == "FC":
        all_outs = net(X)
        outs = all_outs[idx]
    elif model_name == "DHGNN":
        ids = [i for i in range(X.size()[0])]
        ids = torch.tensor(ids).long().to(device)
        all_outs = net(ids=ids, feats=X, edge_dict=A.e_list, G=A.H, ite=epoch, device=device)
        outs = all_outs[idx]
    else:
        all_outs = net(X, A)
        outs = all_outs[idx]

    lbls = lbls[idx]
    lbls = torch.argmax(lbls, dim=1)
    outs = torch.argmax(outs, dim=1)
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res, all_outs


def select_model(feat_dimension, n_hidden_layers, n_classes, n_conv, model, drop_rate, he_dropout, adjacent_centers, clusters, k_structured, k_nearest, k_cluster, wu_kmeans, wu_struct):
        if model == "HGNN":
            return HGNN(feat_dimension, n_hidden_layers, n_classes, n_conv, use_bn=True, drop_rate=drop_rate, he_dropout=he_dropout)
        elif model == "HGNNP":
            return HGNNP(feat_dimension, n_hidden_layers, n_classes, use_bn=True, drop_rate=drop_rate, he_dropout=he_dropout)
        elif model == "FC":
            return FC(feat_dimension, n_classes)
        elif model == "DHGNN":
            n_layers = 2
            return DHGNN(dim_feat=feat_dimension,
            n_categories=n_classes,
            k_structured=k_structured,
            k_nearest=k_nearest,
            k_cluster=k_cluster,
            wu_knn=0,
            wu_kmeans=wu_kmeans,
            wu_struct=wu_struct,
            clusters=clusters,
            adjacent_centers=adjacent_centers,
            n_layers=n_layers,
            layer_spec=[feat_dimension for l in range(n_layers-1)],
            dropout_rate=drop_rate,
            has_bias=True,
            )
        
def structure_builder(trial):

    G = Hypergraph(n_nodes)
    k = trial.suggest_int("k", 3, 100)
    G.add_hyperedges_from_feature_kNN(X, k=k)

    if use_attributes:
        for a in sa:
            G.add_hyperedges(a)
        for a in va:
            G.add_hyperedges(a)
        i = 0
        for a in lpa:
            G.add_hyperedges(a)
            i += 1

        i = 0
        for a in hpa:
            G.add_hyperedges(a)
            i += 1


    G.to(device)
    return G


def model_builder(trial):
    n_layers = 2
    return DHGNN(dim_feat=dim_features,
            n_categories=n_classes,
            k_structured=trial.suggest_int("k_structured", 3, 100),
            k_nearest=trial.suggest_int("k_nearest", 3, 100),
            k_cluster=trial.suggest_int("k_cluster", 3, 100),
            wu_knn=0,
            wu_kmeans=trial.suggest_int("wu_kmeans", 0, 15),
            wu_struct=trial.suggest_int("wu_struct", 0, 15),
            clusters=trial.suggest_int("clusters", 100, 1000),
            adjacent_centers=trial.suggest_int("adjacent_centers", 1, 5),
            n_layers=n_layers,
            layer_spec=[dim_features for l in range(n_layers-1)],
            dropout_rate=trial.suggest_float("drop_rate", 0, 0.9),
            has_bias=True,
            )
    # return HGNNP(dim_features, trial.suggest_int("hidden_dim", 2, 50), num_classes, num_conv=trial.suggest_int("n_conv", 2, 8), use_bn=True, drop_rate=trial.suggest_float("drop_rate", 0, 0.9), he_dropout=trial.suggest_float("he_dropout", 0, 0.9)).to(device)


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

if __name__ == "__main__":
    # set_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # selected_modalities = [['ECG'], ['EEG'], ['EMO'], ['GSR']]
    # selected_modalities = [['GSR']]
    # selected_modalities = [['ECG', 'EMO']]
    selected_modalities = [['ECG', 'EEG', 'EMO', 'GSR']]
    # selected_modalities=[[['ECG'], ['EEG'], ['EMO'], ['GSR'], ['ECG', 'EEG'], ['ECG', 'EMO'], ['ECG', 'GSR'], ['EEG', 'EMO'], ['EEG', 'GSR'], ['EMO', 'GSR'], ['ECG', 'EEG', 'EMO'], ['ECG', 'EEG', 'GSR'], ['ECG', 'EMO', 'GSR'], ['EEG', 'EMO', 'GSR'], ['ECG', 'EEG', 'EMO', 'GSR']]]
    # selected_modalities=[['ECG'], ['EEG'], ['EMO'], ['GSR'], ['ECG', 'EEG'], ['ECG', 'EMO'], ['ECG', 'GSR'], ['EEG', 'EMO'], ['EEG', 'GSR'], ['EMO', 'GSR'], ['ECG', 'EEG', 'EMO'], ['ECG', 'EEG', 'GSR'], ['ECG', 'EMO', 'GSR'], ['EEG', 'EMO', 'GSR'], ['ECG', 'EEG', 'EMO', 'GSR']]


    label = "valence"
    train_ratio = 70
    val_ratio = 15
    test_ratio = 15
    n_classes = 2
    n_hidden_layers = 8 #8
    k = 95 #4, 20    
    lr = 0.0072 #0.01, 0.001
    weight_decay = 0.0076
    n_conv = 2
    drop_rate = 0.03
    he_dropout = 0
    n_epoch = 10000
    model_name = "DHGNN" #HGNN, HGNNP, NB, SVM
    fusion_model = "HGNNP"
    fuse_models = True
    use_attributes = False
    opti = False
    trials = 10

    k = 66 #4, 20   
    drop_rate = 0.37
    lr = 0.001 #0.01, 0.001
    weight_decay = 5e-4
    n_hidden_layers = 8 #8
    n_conv = 2
    he_dropout = 0.5

    # For DHGNN
    d_adjacent_centers=[4,5,2,4]
    d_clusters=[455, 366, 365, 215]
    d_drop_rate = [0.23, 0.37, 0.5, 0.56]
    d_k = [77,91,51,45] 
    d_k_cluster= [88,36,81,13]
    d_k_nearest=[57,56,59,31]
    d_k_structured=[25,88,43,29]
    d_lr = [0.0042,0.0081,0.0029,0.0013] #0.01, 0.001
    d_weight_decay = [0.0073,0.0089,0.0084,0.0099]
    d_wu_kmeans=[3,0,11,13]
    d_wu_struct=[8,9,15,10]

    final_acc = 0
    final_f1 = 0
    all_accs = [0 for m in selected_modalities]
    all_f1s = [0 for m in selected_modalities]

    if opti:
        # work_root = "D:\Dev\THU-HyperG\examples\logs" # PC
        work_root = "/home/adriendutfoy/Desktop/Dev/MultiGraph/examples/logs" # JEMARO computer

        num_classes = 2


        X, Y, train_mask, test_mask, val_mask, sa, va, lpa, hpa = load_ASERTAIN(selected_modalities=selected_modalities[0], label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        print("Optimize for: " + str(selected_modalities[0]))
        dim_features = X.shape[1]
        n_nodes = X.shape[0]
        
        Y = torch.from_numpy(Y).long()
        train_mask = torch.tensor(train_mask).to(device)
        val_mask = torch.tensor(val_mask).to(device)
        test_mask = torch.tensor(test_mask).to(device)
        X = torch.tensor(X).float()
        X = X.to(device)
        Y = Y.to(device)
        input_data = {
            "features": X,
            "labels": Y,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }
        evaluator = Evaluator(["accuracy", "f1_score"])
        task = Task(
            work_root, input_data, model_builder, train_builder, evaluator, device, structure_builder=structure_builder,
        ).to(device)

        task.run(200, 100, "maximize")


    else:
        print_log("model: " + model_name)
        if model_name == "NB" or model_name == "SVM":
            run_baseline(selected_modalities, label, 80, 10, 10, model_name, trials)

        else:
            for trial in range(trials):
                print_log("trial: " + str(trial))
                i = 0
                inputs = []
                accs = []
                for m in selected_modalities:
                    # n_hidden_layers = hds[i]

                    adjacent_centers = d_adjacent_centers[i]
                    clusters = d_clusters[i]
                    drop_rate = d_drop_rate[i]
                    k = d_k[i]
                    k_cluster = d_k_cluster[i]
                    k_nearest = d_k_nearest[i]
                    k_structured = d_k_structured[i]
                    lr = d_lr[i]
                    weight_decay = d_weight_decay[i]
                    wu_kmeans = d_wu_kmeans[i]
                    wu_struct = d_wu_struct[i]


                    print_log("loading data: " + str(m))
                    X, Y, train_mask, test_mask, val_mask, sa, va, lpa, hpa = load_ASERTAIN(selected_modalities=m, label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, trial=trial)
                    model = select_model(feat_dimension=X.shape[1], n_hidden_layers=n_hidden_layers, n_classes=n_classes, model=model_name, n_conv=n_conv, drop_rate=drop_rate, he_dropout=he_dropout, adjacent_centers=adjacent_centers, clusters=clusters, k_cluster=k_cluster, k_nearest=k_nearest, k_structured=k_structured, wu_kmeans=wu_kmeans, wu_struct=wu_struct)

                    X = torch.tensor(X).float()

                    print_log("generating hypergraph: " + str(m))
                    G = Hypergraph(X.size()[0], device=device)
                    G.add_hyperedges_from_feature_kNN(X, k=k)

                    # if use_attributes:
                    #     for a in sa:
                    #         G.add_hyperedges(a, group_name="subject_attributes_"+str(a))
                    #     for a in va:
                    #         G.add_hyperedges(a, group_name="video_attributes_"+str(a))
                    #     z = 0
                    #     for a in lpa:
                    #         G.add_hyperedges(a, group_name="low_personality_attributes_"+str(z))
                    #         z += 1

                    #     z = 0
                    #     for a in hpa:
                    #         G.add_hyperedges(a, group_name="high_personality_attributes_"+str(z))
                    #         z += 1

                    Y = [[0,1] if e == 1 else [1,0] for e in Y]
                    Y = torch.tensor(Y).float()
                    train_mask = torch.tensor(train_mask)
                    val_mask = torch.tensor(val_mask)
                    test_mask = torch.tensor(test_mask)
                    # X = torch.eye(G.num_v)

                    G.to(device)
                    X = X.to(device)
                    Y = Y.to(device)


                    # lr = lrs[i]
                    # weight_decay = wds[i]
                    res, out = run(device, X, Y, train_mask, test_mask, val_mask, G, model, lr , weight_decay, n_epoch, model_name)
                    all_accs[i] += res['accuracy']
                    all_f1s[i] += res['f1_score']
                    accs.append(res['accuracy'])
                    inputs.append(out)
                    i += 1

                if fuse_models:
                    print_log("fusing models with: " + fusion_model)

                    k = 4 #4, 20   
                    drop_rate = 0.5
                    lr = 0.001 #0.01, 0.001
                    weight_decay = 5*10**-4
                    n_hidden_layers = 8 #8
                    n_conv = 2
                    he_dropout = 0.5

                    if fusion_model=="HGNNP":
                        G = Hypergraph(2088)
                        i = 0

                        # weight of attributes 
                        accs.append(0.5)
                        # normalize weights so their sum is 1
                        weights = [float(i)/sum(accs) for i in accs]
                        print("weights: ", weights)
                        average_weight_index = len(inputs)


                        if use_attributes:
                            for a in sa:
                                G.add_hyperedges(a, group_name="attr", e_weight=weights[average_weight_index])
                           
                            for a in va:
                                G.add_hyperedges(a, group_name="attr", e_weight=weights[average_weight_index])

                            for a in lpa:
                                G.add_hyperedges(a, group_name="attr", e_weight=weights[average_weight_index])
                                i += 1

                            for a in hpa:
                                G.add_hyperedges(a, group_name="attr", e_weight=weights[average_weight_index])

                        j = 0
                        for i in inputs:
                            G.add_hyperedges_from_feature_kNN(i, k=k, group_name="modality_"+str(j), e_weight=weights[j])
                            j += 1

                        inputs = torch.cat(inputs, 1)
                        G.add_hyperedges_from_feature_kNN(inputs, k=k, group_name="modality_fusion", e_weight=weights[average_weight_index])
                    
                    else:
                        inputs = torch.cat(inputs, 1)

                    model = select_model(feat_dimension=inputs.size()[1], n_hidden_layers=n_hidden_layers, n_classes=n_classes, model=fusion_model, n_conv=n_conv, drop_rate=drop_rate, he_dropout=he_dropout, adjacent_centers=adjacent_centers, clusters=clusters, k_cluster=k_cluster, k_nearest=k_nearest, k_structured=k_structured, wu_kmeans=wu_kmeans, wu_struct=wu_struct)

                    final_res, _ = run(device, inputs, Y, train_mask, test_mask, val_mask, G, model, lr , weight_decay, n_epoch, fusion_model)
                    final_acc += final_res['accuracy']
                    final_f1 += final_res['f1_score']


            print("acc: ", np.divide(all_accs,trials))
            print("f1: ", np.divide(all_f1s,trials))
            print(selected_modalities)

            if fuse_models:
                print("final acc: ", final_acc/trials)
                print("final f1: ", final_f1/trials)

    









    
