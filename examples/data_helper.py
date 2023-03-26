import os
import pickle as pkl

import numpy as np
import scipy.io as scio
import SimpleITK as sitk
from sklearn.preprocessing import normalize

from hyperg.utils import minmax_scale
from hyperg.utils import print_log

import csv
from itertools import groupby
import random

# random.seed(0)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'datasets')


def load_myocardium(test_idx=[4]):
    heart_seg_dir = os.path.join(DATA_DIR, 'myocardiumSeg')
    ori = os.listdir(os.path.join(heart_seg_dir, 'ori'))

    X = []
    y = []

    for name in ori:
        ori_img = sitk.ReadImage(os.path.join(heart_seg_dir, "ori/{}".format(name)))
        ori_ary = minmax_scale(sitk.GetArrayFromImage(ori_img).squeeze()) # (y, x)
        X.append(ori_ary)

        seg_img = sitk.ReadImage(os.path.join(heart_seg_dir, "seg/{}".format(name)))
        seg_ary = sitk.GetArrayFromImage(seg_img).squeeze()
        y.append(seg_ary)

    X = np.stack(X)
    y = np.stack(y)

    training_idx = [i for i in range(X.shape[0]) if i not in test_idx]

    X_train = X[training_idx]
    X_test = X[test_idx]
    y_train = y[training_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def load_modelnet(selected_mod):
    print_log("selected mod:{}".format(str(selected_mod)))
    modelnet40_dir = os.path.join(DATA_DIR, "modelnet40")
    X_train = pkl.load(open(os.path.join(modelnet40_dir, 'modelnet_train_fts.pkl'), 'rb'))
    X_test = pkl.load(open(os.path.join(modelnet40_dir, 'modelnet_test_fts.pkl'), 'rb'))

    y_train = pkl.load(open(os.path.join(modelnet40_dir, 'modelnet_train_lbls.pkl'), 'rb'))
    y_test = pkl.load(open(os.path.join(modelnet40_dir, 'modelnet_test_lbls.pkl'), 'rb'))

    X_train = [X_train[imod] for imod in selected_mod]
    X_test = [X_test[imod] for imod in selected_mod]

    if len(selected_mod) == 1:
        X_train = X_train[0]
        X_test = X_test[0]

    return X_train, X_test, np.array(y_train), np.array(y_test)


def load_MSRGesture3D(i_train=2, i_test = 0):
    msr_gesture_dir = os.path.join(DATA_DIR, "MSRGesture3D")
    data = scio.loadmat(os.path.join(msr_gesture_dir, 'MSRGesture3D.mat'))
    all_indices = scio.loadmat(os.path.join(msr_gesture_dir, 'MSRGesture3DTrainIndex.mat'))['trainIndex']

    i_indices = all_indices[i_test, i_train].reshape(-1)
    X = data['X']
    X = normalize(X)
    y = np.array(data['Y'], dtype=np.int).reshape(-1)
    y = y - np.min(y)

    X_train = X[i_indices == 1]
    X_test = X[i_indices == 0]
    y_train = y[i_indices == 1]
    y_test = y[i_indices == 0]

    return X_train, X_test, y_train, y_test

def load_ASERTAIN(selected_modalities=['ECG', 'GSR'], train_ratio=80, label='valence'):

    if label == 'valence':
        label_index = 2
    elif label == 'arousal':
        label_index = 3

    dir = os.path.join(DATA_DIR, "ASCERTAIN_Features")
    """
	convert csv to np array
	"""
    with open(os.path.join(dir, "ascertain_multimodal.csv")) as file:
        reader = csv.reader(file)
        data = list(reader)
        columns = np.asarray(data[0])
        data = np.asarray(data[1:]).astype(float)

    subject_id_index = np.where(columns == 'subject_id')[0][0]
    case_id_index = np.where(columns == 'case_id')[0][0]

    # select modality
    selected_index = [i for i in range(len(columns)) 
							if (not is_column_feature(columns, i)) or (is_column_feature(columns, i) and (columns[i].split('_')[0] in selected_modalities)) ]
    data = data[:,selected_index]
    # columns = [columns[i] for i in selected_index if i < 4]


    """
    split train and test dataset subject-wise based on self.split_ratio
    group data upon subject id
    """
    data_grouped = [list(it) for k, it in groupby(data.tolist())]
    random.shuffle(data_grouped)

    attributes_idx = {"subjects": []}

    for i in range(58):
        subject_id = {'attri_'+str(i): []}
        attributes_idx["subjects"].append(subject_id)

    all_data = [item for sublist in data_grouped for item in sublist]
    for i in range(len(all_data)):
        id = int(all_data[i][0])
        attributes_idx['subjects'][id]['attri_'+str(id)].append(i)

    print(attributes_idx['subjects'])

    split_index = int((len(data_grouped))*train_ratio/100)
    train = data_grouped[:split_index]
    train = [item for sublist in train for item in sublist] # flatten
    test = data_grouped[split_index:]
    test = [item for sublist in test for item in sublist] # flatten
    
    
    X_train = np.asarray(train)[1:, 3:]
    X_test = np.asarray(test)[1:, 3:]
    y_train = np.asarray(train)[1:, label_index]
    y_test = np.asarray(test)[1:, label_index]

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    return X_train, X_test, y_train, y_test


def is_column_feature(columns, column_index):
	return ('label' not in columns[column_index] and 'id' not in columns[column_index])


if __name__ == "__main__":
    pass
