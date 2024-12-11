import os
import warnings
import glob
import csv
import re
import numpy as np
import scipy.io as sio
import sys
from nilearn import connectome
import pandas as pd
from scipy.spatial import distance
from scipy import signal
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# Input data variables

root_folder = './data/'


# Get timeseries arrays for list of subjects
def get_timeseries(subject_list, dataset_name, atlas_name, silence=False):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200
    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

    print('load timeseries...')
    timeseries = []
    data_folder = os.path.join(root_folder, '{}_roi'.format(dataset_name))
    valid_subject_list = []
    for i in range(len(subject_list)):
        if dataset_name == 'adhd200':
            filename = 'fmri_X_{:07}_session_1_run1.nii_aal3.npy'.format(int(subject_list[i]))
        elif dataset_name == 'cobre':
            filename = 'sub-{}_ses-20110101_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii_aal3.npy'.format(subject_list[i])
        elif dataset_name == 'UCLA':
            filename = '{}_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii_ho.npy'.format(subject_list[i])
        elif dataset_name == 'hcp-ep':
            filename = 'sub-{}_task-rest_acq-pa_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii_aal3.npy'.format(subject_list[i])
        else:
            import ipdb; ipdb.set_trace()

        if os.path.exists(os.path.join(data_folder, filename)):
            valid_subject_list.append(subject_list[i])
            data = np.load(os.path.join(data_folder, filename))
            timeseries.append(data.T)

    return timeseries, valid_subject_list


#  compute connectivity matrices
def subject_connectivity(timeseries, subjects, atlas_name, kind, iter_no='', seed=1234, n_subjects='', save=True, save_path=None):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subjects     : subject IDs
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        iter_no      : tangent connectivity iteration number for cross validation evaluation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder
    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    if kind in ['correlation', 'partial correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform(timeseries)
    
    if save and not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        for i, subj_id in enumerate(subjects):
            os.makedirs(os.path.join(save_path, subj_id), exist_ok=True)
            subject_file = os.path.join(save_path, subj_id, subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
            sio.savemat(subject_file, {'connectivity': connectivity[i]})
        
        return connectivity


def get_subject_score(subject_list, dataset_name, score):
    dataset_name = dataset_name.lower()
    file_path = './data/{}-rest.csv'.format(dataset_name)
    scores_dict = {}
    all_scores = {}
    
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            subject_id = row['subject_id']
            try:
                if score == 'DX':
                    score_value = int(row[score])
                elif score in ['Gender', 'sex', 'gender']:
                    if dataset_name == 'adhd200':
                        score_value = int(float(row[score]))
                    elif dataset_name == 'cobre':
                        score_value = 0 if row[score] == 'male' else 1
                    elif dataset_name == 'UCLA':
                        score_value = 0 if row[score] == 'M' else 1
                    elif dataset_name == 'hcp-ep':
                        score_value = 0 if row[score] == 'M' else 1
                else:
                    import ipdb; ipdb.set_trace()
            except:
                score_value = -1
            all_scores[subject_id] = score_value
    
    for subject in subject_list:
        subject_id = str(subject)
        if subject_id in all_scores:
            scores_dict[subject_id] = all_scores[subject_id]

    return scores_dict


# preprocess phenotypes. Categorical -> ordinal representation
def preprocess_phenotypes(pheno_ft, params):
    if params['model'] == 'MIDA':
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2])], remainder='passthrough')
    else:
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2, 3])], remainder='passthrough')

    pheno_ft = ct.fit_transform(pheno_ft)
    pheno_ft = pheno_ft.astype('float32')

    return (pheno_ft)


# create phenotype feature vector to concatenate with fmri feature vectors
def phenotype_ft_vector(pheno_ft, num_subjects, params):
    gender = pheno_ft[:, 0]
    if params['model'] == 'MIDA':
        eye = pheno_ft[:, 0]
        hand = pheno_ft[:, 2]
        age = pheno_ft[:, 3]
        fiq = pheno_ft[:, 4]
    else:
        eye = pheno_ft[:, 2]
        hand = pheno_ft[:, 3]
        age = pheno_ft[:, 4]
        fiq = pheno_ft[:, 5]

    phenotype_ft = np.zeros((num_subjects, 4))
    phenotype_ft_eye = np.zeros((num_subjects, 2))
    phenotype_ft_hand = np.zeros((num_subjects, 3))

    for i in range(num_subjects):
        phenotype_ft[i, int(gender[i])] = 1
        phenotype_ft[i, -2] = age[i]
        phenotype_ft[i, -1] = fiq[i]
        phenotype_ft_eye[i, int(eye[i])] = 1
        phenotype_ft_hand[i, int(hand[i])] = 1

    if params['model'] == 'MIDA':
        phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand], axis=1)
    else:
        phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand, phenotype_ft_eye], axis=1)

    return phenotype_ft


# Load precomputed fMRI connectivity networks
def get_networks(subject_list, kind, save_path, iter_no='', seed=1234, n_subjects='', atlas_name="aal", variable='connectivity'):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks
    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    all_networks = []
    for subject in subject_list:
        if len(kind.split()) == 2:
            kind = '_'.join(kind.split())
        fl = os.path.join(save_path, subject, subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")

        matrix = sio.loadmat(fl)[variable]
        all_networks.append(matrix)

    norm_networks = [np.arctanh(mat) for mat in all_networks]
    networks = np.stack(norm_networks)

    return networks

