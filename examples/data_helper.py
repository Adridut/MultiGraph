import os
import pickle as pkl

import numpy as np
import scipy.io as scio
import SimpleITK as sitk
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import csv
from itertools import groupby
import random

from sklearn.impute import SimpleImputer
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler



DATA_DIR = os.path.join(os.path.dirname(__file__), 'datasets')

def load_ASERTAIN(selected_modalities=['ECG', 'GSR'],  label='valence', train_ratio=60, val_ratio=20, test_ratio=20, trial=0):

    # random.seed(trial)
    
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

    # for i in range(n_subjects):
    #     subject_id = []
    #     subject_attributes.append(subject_id)

    # for i in range(n_cases):
    #     video_id = []
    #     video_attributes.append(video_id)

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
        # subject_attributes[subject_id].append(i)
        case_id = int(all_data[i][1])
        # video_attributes[case_id].append(i)
        for j in range(len(low_personality_attributes)):
            personality_trait = all_data[i][all_data.shape[1]-j-1]
            if j == 0 or j == 3:
                threshold = 4
            else:
                threshold = 5
            if personality_trait < threshold:
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


    train_mask = [True for i in range(round(len(X)*train_ratio/100))] + [False for i in range(round(len(X)-len(X)*train_ratio/100))]
    test_mask = [False for i in range(round(len(X) - len(X)*test_ratio/100))] + [True for i in range(round(len(X)*test_ratio/100))]
    valid_mask =  np.logical_and(np.logical_not(train_mask),  np.logical_not(test_mask))


    # Fill nan values
    
    scaler = StandardScaler()
    impNan = SimpleImputer(missing_values=0, strategy='mean')
    impInf = SimpleImputer(missing_values=np.inf, strategy='mean')
    

    # X = KNN(k=3).fit_transform(X, y)

    X = np.nan_to_num(X)

    X = normalize(X)

    X[train_mask] = impInf.fit_transform(X[train_mask], y[train_mask])
    X[valid_mask] = impInf.transform(X[valid_mask])
    X[test_mask] = impInf.transform(X[test_mask])

    X[train_mask] = impNan.fit_transform(X[train_mask], y[train_mask])
    X[valid_mask] = impNan.transform(X[valid_mask])
    X[test_mask] = impNan.transform(X[test_mask])

    X[train_mask] = scaler.fit_transform(X[train_mask], y[train_mask])
    X[valid_mask] = scaler.transform(X[valid_mask])
    X[test_mask] = scaler.transform(X[test_mask])










    # X[train_mask] = lda.fit_transform(X[train_mask], y[train_mask])
    # X[valid_mask] = lda.transform(X[valid_mask])
    # X[test_mask] = lda.transform(X[test_mask])


    return X, y, train_mask, test_mask, valid_mask, subject_attributes, video_attributes, low_personality_attributes, high_personality_attributes


def is_column_feature(columns, column_index):
	return ('label' not in columns[column_index] and 'id' not in columns[column_index])


if __name__ == "__main__":
    pass
