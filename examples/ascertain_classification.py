# coding=utf-8
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from hyperg.generation import gen_knn_hg
from hyperg.learning import trans_infer
from hyperg.utils import print_log

from data_helper import load_ASERTAIN


def main():
    print_log("loading data")
    X_train, X_test, y_train, y_test = load_ASERTAIN()

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, -1 * np.ones_like(y_test)])

    print_log("generating hypergraph")
    hg = gen_knn_hg(X, n_neighbors=25)

    print_log("learning on hypergraph")
    y_predict = trans_infer(hg, y, lbd=100)
    print_log("accuracy: {}".format(accuracy_score(y_test, y_predict)))
    print_log("f1: {}".format(f1_score(y_test, y_predict), average='weighted'))



if __name__ == "__main__":
    main()