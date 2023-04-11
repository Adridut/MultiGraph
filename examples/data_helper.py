import os
import pickle as pkl

import numpy as np
import scipy.io as scio
import SimpleITK as sitk
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

def load_ASERTAIN(selected_modalities=['ECG', 'GSR'],  label='valence', train_ratio=60, val_ratio=20, test_ratio=20):

    # random.seed(0)
    
    n_subjects = 58
    n_cases = 36

    if label == 'valence':
        label_index = 3
    elif label == 'arousal':
        label_index = 2

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
    selected_modalities.append('Personality')
    
    selected_index = [i for i in range(len(columns)) 
							if (not is_column_feature(columns, i)) or ((is_column_feature(columns, i) and (columns[i].split('_')[0] in selected_modalities))) ]
    
    data = data[:,selected_index]
    # columns = [columns[i] for i in selected_index if i < 4]


    """
    split train and test dataset subject-wise based on self.split_ratio
    group data upon subject id
    """
    data_grouped = [list(it) for k, it in groupby(data.tolist())]
    random.shuffle(data_grouped)


    subject_attributes = []
    video_attributes = []
    low_personality_attributes = []
    high_personality_attributes = []

    for i in range(n_subjects):
        subject_id = []
        subject_attributes.append(subject_id)

    for i in range(n_cases):
        video_id = []
        video_attributes.append(video_id)

    # 5 = number of personality traits
    for i in range(5):
        personality_id = []
        low_personality_attributes.append(personality_id)

    for i in range(5):
        personality_id = []
        high_personality_attributes.append(personality_id)


    all_data = [item for sublist in data_grouped for item in sublist]
    all_data = np.asarray(all_data)
    

    for i in range(len(all_data[0:, 4:])):
        subject_id = int(all_data[i][0])
        subject_attributes[subject_id].append(i)
        case_id = int(all_data[i][1])
        video_attributes[case_id].append(i)
        for j in range(len(low_personality_attributes)):
            personality_trait = all_data[i][all_data.shape[1]-j-1]
            if personality_trait < 5:
                low_personality_attributes[j].append(i)
            else:
                high_personality_attributes[j].append(i)


    selected_modalities.remove('Personality')
    # split_index = int((len(data_grouped))*train_ratio/100)
    # train = data_grouped[:split_index+1]
    # train = [item for sublist in train for item in sublist] # flatten
    # test = data_grouped[split_index:]
    # test = [item for sublist in test for item in sublist] # flatten
        
    # remove ids, labels and personalities from features
    X = all_data[0:, 4:all_data.shape[1]-5]
    y = all_data[0:, label_index]

    X = normalize(X)

    train_mask = [True for i in range(round(len(X)*train_ratio/100))] + [False for i in range(round(len(X)-len(X)*train_ratio/100))]
    test_mask = [False for i in range(round(len(X) - len(X)*test_ratio/100))] + [True for i in range(round(len(X)*test_ratio/100))]
    valid_mask =  np.logical_and(np.logical_not(train_mask),  np.logical_not(test_mask))

    # scaler = StandardScaler()
    # lda = LinearDiscriminantAnalysis()

    # X[train_mask] = scaler.fit_transform(X[train_mask], y[train_mask])
    # X[valid_mask] = scaler.fit_transform(X[valid_mask])
    # X[test_mask] = scaler.fit_transform(X[test_mask])

    # X[train_mask] = lda.fit_transform(X[train_mask], y[train_mask])
    # X[valid_mask] = lda.transform(X[valid_mask])
    # X[test_mask] = lda.transform(X[test_mask])


    return X, y, train_mask, test_mask, valid_mask, subject_attributes, video_attributes, low_personality_attributes, high_personality_attributes


def is_column_feature(columns, column_index):
	return ('label' not in columns[column_index] and 'id' not in columns[column_index])


if __name__ == "__main__":
    pass
