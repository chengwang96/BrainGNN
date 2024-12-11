import argparse
from imports import preprocess_data as Reader
import os
import csv
import warnings

warnings.filterwarnings("ignore")


def load_subject_id(dataset_name):
    subject_ids = []
    dataset_name = dataset_name.lower()
    file_path = './data/{}-rest.csv'.format(dataset_name)
    
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        
        for row in csvreader:
            if row: 
                subject_ids.append(row[0])
    
    return subject_ids


def main():
    parser = argparse.ArgumentParser(description='Download ABIDE data and compute functional connectivity matrices')
    parser.add_argument('--atlas', default='aal3', help='aal3 dk cc200 ho')
    parser.add_argument('--dataset_name', default='adhd200', help='adhd200, cobre, hcp-d, hcp-ep, UCLA, ABCD')
    parser.add_argument('--dataset_dir', default='data')
    args = parser.parse_args()
    print(args)

    subject_IDs = load_subject_id(args.dataset_name)
    time_series, valid_subject_list = Reader.get_timeseries(subject_IDs, args.dataset_name, args.atlas)

    # Compute and save connectivity matrices
    save_path = os.path.join(args.dataset_dir, '{}_roi'.format(args.dataset_name), 'braingnn_{}'.format(args.atlas))
    Reader.subject_connectivity(time_series, valid_subject_list, args.atlas, 'correlation', save_path=save_path)
    Reader.subject_connectivity(time_series, valid_subject_list, args.atlas, 'partial correlation', save_path=save_path)

    import pickle
    pkl_name = os.path.join(save_path, 'valid_subject_list.pkl')
    with open(pkl_name, 'wb') as file:
        pickle.dump(valid_subject_list, file)


if __name__ == '__main__':
    main()