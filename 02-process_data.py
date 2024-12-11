import pickle
import argparse
import pandas as pd
import numpy as np
from imports import preprocess_data as Reader
import deepdish as dd
import warnings
import os
import tqdm

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a Ridge classifier. MIDA is used to minimize the distribution mismatch between ABIDE sites')
    parser.add_argument('--atlas', default='aal3', help='aal3 dk cc200 ho')
    parser.add_argument('--score', default='DX', help='DX Gender Age')
    parser.add_argument('--dataset_name', default='adhd200', help='adhd200, cobre, hcp-d, hcp-ep, UCLA, ABCD')
    parser.add_argument('--dataset_dir', default='data')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation. default: 1234.')
    parser.add_argument('--nclass', default=2, type=int, help='Number of classes. default:2')

    args = parser.parse_args()
    print('Arguments: \n', args)

    params = dict()
    params['seed'] = args.seed  # seed for random initialisation

    # Algorithm choice
    params['atlas'] = args.atlas  # Atlas for network construction
    atlas = args.atlas  # Atlas for network construction (node definition)

    save_path = os.path.join(args.dataset_dir, '{}_roi'.format(args.dataset_name), 'braingnn_{}'.format(args.atlas))
    pkl_name = os.path.join(save_path, 'valid_subject_list.pkl')
    with open(pkl_name, 'rb') as file:
        subject_IDs = pickle.load(file)
    
    labels = Reader.get_subject_score(subject_IDs, args.dataset_name, score=args.score)

    # Number of subjects and classes for binary classification
    num_classes = args.nclass
    num_subjects = len(subject_IDs)
    params['n_subjects'] = num_subjects

    # Initialise variables for class labels and acquisition sites
    # 1 is autism, 2 is control
    y = np.zeros([num_subjects, 1]) # n x 1

    # Get class labels for all subjects
    for i in range(num_subjects):
        y[i] = labels[subject_IDs[i]]

    # Compute feature vectors (vectorised connectivity networks)
    fea_corr = Reader.get_networks(subject_IDs, iter_no='', kind='correlation', save_path=save_path, atlas_name=atlas)
    fea_pcorr = Reader.get_networks(subject_IDs, iter_no='', kind='partial correlation', save_path=save_path, atlas_name=atlas)

    if not os.path.exists(os.path.join(save_path, 'raw')):
        os.makedirs(os.path.join(save_path, 'raw'))

    tbar = tqdm.tqdm(total=len(subject_IDs))
    for i, subject in enumerate(subject_IDs):
        dd.io.save(os.path.join(save_path, 'raw', subject + '.h5'), {'corr': fea_corr[i], 'pcorr': fea_pcorr[i], 'label': y[i]})
        tbar.update(1)
    tbar.close()

if __name__ == '__main__':
    main()
