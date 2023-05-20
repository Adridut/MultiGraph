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
    loss = F.binary_cross_entropy(outs, lbls)
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
    lbls = torch.argmax(lbls, dim=1)
    outs = torch.argmax(outs, dim=1)
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res, all_outs


def select_model(feat_dimension, n_hidden_layers, n_classes, n_conv, model):
        if model == "HGNN":
            return HGNN(feat_dimension, n_hidden_layers, n_classes, n_conv, use_bn=True)
        elif model == "HGNNP":
            return HGNNP(feat_dimension, n_hidden_layers, n_classes, use_bn=True)
        elif model == "FC":
            return FC(feat_dimension, n_classes)
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
        
def structure_builder(trial):

    G = Hypergraph(2088)
    for m in selected_modalities:
        for mod in m:
            x, y, train_mask, test_mask, val_mask, sa, va, lpa, hpa = load_ASERTAIN(selected_modalities=[mod], label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
            x = torch.tensor(x).float()
            k = trial.suggest_int("k_"+str(mod), 3, 100)
            G.add_hyperedges_from_feature_kNN(x, k=k, group_name=str(mod))

    if use_attributes:
        # for a in sa:
        #     G.add_hyperedges(a, group_name="subject_attributes_"+str(a))
        # for a in va:
        #     G.add_hyperedges(a, group_name="video_attributes_"+str(a))
        i = 0
        for a in lpa:
            G.add_hyperedges(a, group_name="low_personality_attributes_"+str(i))
            i += 1

        i = 0
        for a in hpa:
            G.add_hyperedges(a, group_name="high_personality_attributes_"+str(i))
            i += 1


    G.to(device)
    return G


def model_builder(trial):
    return HGNNP(dim_features, trial.suggest_int("hidden_dim", 2, 50), num_classes, use_bn=True, drop_rate=trial.suggest_float("drop_rate", 0, 0.9)).to(device)


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
    selected_modalities = [['ECG'], ['EEG'], ['EMO'], ['GSR']]
    # selected_modalities = [['EEG']]
    # selected_modalities = [['ECG', 'EMO']]
    # selected_modalities = [['ECG', 'EEG', 'EMO', 'GSR']]
    # selected_modalities=[[['ECG'], ['EEG'], ['EMO'], ['GSR'], ['ECG', 'EEG'], ['ECG', 'EMO'], ['ECG', 'GSR'], ['EEG', 'EMO'], ['EEG', 'GSR'], ['EMO', 'GSR'], ['ECG', 'EEG', 'EMO'], ['ECG', 'EEG', 'GSR'], ['ECG', 'EMO', 'GSR'], ['EEG', 'EMO', 'GSR'], ['ECG', 'EEG', 'EMO', 'GSR']]]
    # selected_modalities=[['ECG'], ['EEG'], ['EMO'], ['GSR'], ['ECG', 'EEG'], ['ECG', 'EMO'], ['ECG', 'GSR'], ['EEG', 'EMO'], ['EEG', 'GSR'], ['EMO', 'GSR'], ['ECG', 'EEG', 'EMO'], ['ECG', 'EEG', 'GSR'], ['ECG', 'EMO', 'GSR'], ['EEG', 'EMO', 'GSR'], ['ECG', 'EEG', 'EMO', 'GSR']]

    # ks = [[58], [22], [45], [14], [49, 99], [85,43]]
    ks = [28, 95, 28, 18, 6, 80, 95, 91, 51, 51, 19, 69, 20, 62, 9]
    lrs = [0.00013869861245357332, 0.0011044005450656853, 0.005636798360593478, 0.00941688905278987, 0.0006189295525337117, 0.0026484233717115353]
    wds = [0.0001493683554419846, 0.006182977400901223, 0.004364870880569334, 0.006812298690059751, 0.0004035446353541675, 0.00014742828620655684]
    hds = [19, 8, 13, 15, 2, 11]

    label = "arousal"
    train_ratio = 70
    val_ratio = 15
    test_ratio = 15
    n_classes = 2
    n_hidden_layers = 8 #8
    k = 4 #4, 20    
    lr = 0.001 #0.01, 0.001
    weight_decay = 5*10**-4 
    n_conv = 2
    n_epoch = 600
    model_name = "HGNN" #HGNN, HGNNP, NB, SVM
    fusion_model = "HGNNP"
    fuse_models = True
    use_attributes = False
    opti = False
    trials = 1


    final_acc = 0
    final_f1 = 0
    all_accs = [0 for m in selected_modalities]
    all_f1s = [0 for m in selected_modalities]

    if opti:
        work_root = "D:\Dev\THU-HyperG\examples\logs" # PC
        # work_root = "/home/adriendutfoy/Desktop/Dev/MultiGraph/examples/logs" # JEMARO computer

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        num_classes = 2


        X, y, train_mask, test_mask, val_mask, sa, va, lpa, hpa = load_ASERTAIN(selected_modalities=selected_modalities[0], label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        dim_features = X.shape[1]
        
        y = torch.from_numpy(y).long()
        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)
        X = torch.tensor(X).float()
        X = X.to(device)
        y = y.to(device)
        # hg_base = Hypergraph(data["num_vertices"], data["edge_list"])
        input_data = {
            "features": X,
            "labels": y,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "storage": "sqlite:///db.sqlite3"
        }
        evaluator = Evaluator(["accuracy", "f1_score"])
        task = Task(
            work_root, input_data, model_builder, train_builder, evaluator, device, structure_builder=structure_builder,
        ).to(device)

        task.run(n_epoch, 500, "maximize")


    else:
        if model_name == "NB" or model_name == "SVM":
            run_baseline(selected_modalities, label, train_ratio, val_ratio, test_ratio, model_name, trials)

        else:
            for trial in range(trials):
                print_log("trial: " + str(trial))
                i = 0
                inputs = []
                accs = []
                for m in selected_modalities:
                    # n_hidden_layers = hds[i]

                    print_log("loading data: " + str(m))
                    X, Y, train_mask, test_mask, val_mask, sa, va, lpa, hpa = load_ASERTAIN(selected_modalities=m, label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, trial=trial)
                    model = select_model(feat_dimension=X.shape[1], n_hidden_layers=n_hidden_layers, n_classes=n_classes, model=model_name, n_conv=n_conv)

                    X = torch.tensor(X, requires_grad=True).float()

                    print_log("generating hypergraph: " + str(m))
                    G = Hypergraph(X.size()[0])

                    j = 0
                    for mod in m:
                        x, _, _, _, _, _, _, _, _ = load_ASERTAIN(selected_modalities=[mod], label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, trial=trial)
                        x = torch.tensor(x).float()
                        # k = ks[i]
                        G.add_hyperedges_from_feature_kNN(x, k=k, group_name=str(mod))
                        j += 1

                    if use_attributes:
                        for a in sa:
                            G.add_hyperedges(a, group_name="subject_attributes_"+str(a))
                        for a in va:
                            G.add_hyperedges(a, group_name="video_attributes_"+str(a))
                        z = 0
                        for a in lpa:
                            G.add_hyperedges(a, group_name="low_personality_attributes_"+str(z))
                            z += 1

                        z = 0
                        for a in hpa:
                            G.add_hyperedges(a, group_name="high_personality_attributes_"+str(z))
                            z += 1

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
                    print_log("fusing models")

                    if fusion_model=="HGNNP":
                        G = Hypergraph(2088)
                        i = 0

                        for a in sa:
                            G.add_hyperedges(a, group_name="subject_attributes_")
                        for a in va:
                            G.add_hyperedges(a, group_name="video_attributes_")

                        for a in lpa:
                            G.add_hyperedges(a, group_name="low_personality_attributes_")
                            i += 1

                        i = 0
                        for a in hpa:
                            G.add_hyperedges(a, group_name="high_personality_attributes_")
                            i += 1

                        j = 0
                        for i in inputs:
                            G.add_hyperedges_from_feature_kNN(i, k=k, group_name="modality_"+str(j), e_weight=accs[j])
                            j += 1

                        inputs = torch.cat(inputs, 1)
                        G.add_hyperedges_from_feature_kNN(inputs, k=k, group_name="modality_fusion")
                    
                    else:
                        inputs = torch.cat(inputs, 1)

                    model = select_model(feat_dimension=inputs.size()[1], n_hidden_layers=n_hidden_layers, n_classes=n_classes, model=fusion_model, n_conv=n_conv)

                    final_res, _ = run(device, inputs, Y, train_mask, test_mask, val_mask, G, model, lr , weight_decay, n_epoch, fusion_model)
                    final_acc += final_res['accuracy']
                    final_f1 += final_res['f1_score']


            print("acc: ", np.divide(all_accs,trials))
            print("f1: ", np.divide(all_f1s,trials))
            print(selected_modalities)

            if fuse_models:
                print("final acc: ", final_acc/trials)
                print("final f1: ", final_f1/trials)

    









    
