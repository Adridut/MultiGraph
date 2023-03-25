# coding=utf-8
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from hyperg.generation import gen_knn_hg
from hyperg.learning import trans_infer, multi_hg_trans_infer, multi_hg_weighting_trans_infer, tensor_hg_trans_infer
from hyperg.utils import print_log

from data_helper import load_ASERTAIN

from hyperg.learning import multi_hg_trans_infer


def singleHG():
    print_log("loading data")
    selected_modalities=['EMO']
    X_train, X_test, y_train, y_test = load_ASERTAIN(selected_modalities=selected_modalities, label = 'valence', train_ratio=80)

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, -1 * np.ones_like(y_test)])

    print_log("generating hypergraph")
    hg = gen_knn_hg(X, n_neighbors=25)

    print_log("learning on hypergraph")
    y_predict = trans_infer(hg, y, lbd=100)
    print_log("accuracy: {}".format(accuracy_score(y_test, y_predict)))
    print_log("f1: {}".format(f1_score(y_test, y_predict), average='weighted'))


def main():
    print_log("loading data")

    selected_modalities=[['EEG'], ['GSR'], ['ECG'], ['EMO'], ['EEG', 'GSR'], ['EEG', 'ECG'], ['EEG', 'EMO'], ['GSR', 'ECG'], ['GSR', 'EMO'], ['ECG', 'EMO'], ['EEG', 'GSR', 'ECG'], ['EEG', 'GSR', 'EMO'], ['EEG', 'ECG', 'EMO'], ['GSR', 'ECG', 'EMO'], ['EEG', 'GSR', 'ECG', 'EMO']]
    all_X_train = []
    all_X_test = []
    for m in selected_modalities:
        X_train, X_test, y_train, y_test = load_ASERTAIN(selected_modalities=m, label = 'arousal', train_ratio=80)
        all_X_train.append(X_train)
        all_X_test.append(X_test)
    

    X = [np.vstack((all_X_train[imod], all_X_test[imod])) for imod in range(len(selected_modalities))]
    y = np.concatenate((y_train, -1 * np.ones_like(y_test)))

    print_log("generating hypergraph")
    hg_list = [
        gen_knn_hg(X[imod], n_neighbors=25)
        for imod in range(len(selected_modalities))
    ]

    print_log("learning on hypergraph")
    y_predict = multi_hg_weighting_trans_infer(hg_list, y, lbd=100, max_iter=100, mu=0.00000001)
    print_log("accuracy: {}".format(accuracy_score(y_test, y_predict)))
    print_log("f1: {}".format(f1_score(y_test, y_predict), average='weighted'))


def is_column_feature(columns, column_index):
    print(columns, column_index)
    return ('label' not in columns[column_index] and 'id' not in columns[column_index])


if __name__ == "__main__":
    main()