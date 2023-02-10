import os
import json
import argparse
import openml

import pandas as pd

from os import listdir
from os.path import isfile, join
from matplotlib import gridspec

def get_dataset(name):
    loader = {
        'blood': 1464,
        'breast': 1465, #this is breast-tissue, not breast cancer
        'diabetes': 37,
        'ecoli': 40671,
        'iris': 61,
        'parkinsons': 1488,
        'seeds': 1499,
        'thyroid': 40682,
        'vehicle': 54,
        'wine': 187,
    }
    if name in loader:
        return load_dataset_from_openml(loader[name])
    else:
        return load_dataset_from_csv(name)

def load_dataset_from_csv(name):
    data = pd.read_csv(os.path.join('datasets', name +'.csv'), header=None)
    data = data.to_numpy()
    if name == 'parkinsons':
        features_name = [
            'MDVP:Fo(Hz)',
            'MDVP:Fhi(Hz)',
            'MDVP:Flo(Hz)',
            'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)',
            'MDVP:RAP',
            'MDVP:PPQ',
            'Jitter:DDP',
            'MDVP:Shimmer',
            'MDVP:Shimmer(dB)',
            'Shimmer:APQ3',
            'Shimmer:APQ5',
            'MDVP:APQ',
            'Shimmer:DDA',
            'NHR',
            'HNR',
            'RPDE',
            'DFA',
            'spread1',
            'spread2',
            'D2',
            'PPE'
            ]
    elif name == 'seeds':
        features_name = [
            'area',
            'perimeter',
            'compactness',
            'length of kernel',
            'width of kernel',
            'asymmetry coefficient',
            'length of kernel groove'
            ]
    elif name == 'synthetic':
        features_name = [
            'feature_0',
            'feature_1',
            'feature_2',
            'feature_3',
            'feature_4',
        ]
    elif name == 'iris2' or name == 'iris3':
        features_name = [
            '[2 - petallength (numeric)]',
            '[3 - petalwidth (numeric)]',
        ]
    else:
        raise Exception('No features names assigned')
    X, y = data[:, :-1], data[:, -1]
    categorical_indicator = [False for _ in range(X.shape[1])]
    return X, y, features_name

def load_dataset_from_openml(id):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )
    dataset_features_names = [str(elem) for elem in list(dataset.features.values())]
    dataset_features_names = dataset_features_names[:-1]
    return X, y, dataset_features_names

def load_result(input_path, dataset, metric):
    results = pd.DataFrame()
    file_name =  dataset + '_' + metric[0] + '.json'
    _, _, original_features = get_dataset(dataset)
    num_features = len(original_features)
    tot_conf = (44 * num_features) if metric[2] == 'toy' else (2310 * (1 + 4*(num_features-1)))
    with open(os.path.join(input_path, file_name)) as json_file:
        data = json.load(json_file)
        history = data['context']['history']
        for elem in history:
            results = results.append(pd.DataFrame({
                'dataset': [dataset],
                'iteration': [elem['iteration']], 
                #'pipeline': [elem['pipeline']], 
                #'algorithm': [elem['algorithm']], 
                'features': ['None' if elem['pipeline']['features'][0] == 'features_NoneType' else elem['pipeline']['features'][0]], 
                'features__k': ['None' if elem['pipeline']['features'][0] == 'features_NoneType' else elem['pipeline']['features'][1]['features__k']], 
                'normalize': ['None' if elem['pipeline']['normalize'][0] == 'normalize_NoneType' else elem['pipeline']['normalize'][0]], 
                #'normalize__with_mean': ['None' if (elem['pipeline']['normalize'][0] == 'normalize_NoneType' or elem['pipeline']['normalize'][0] == 'normalize_MinMaxScaler') else elem['pipeline']['normalize'][1]['normalize__with_mean']], 
                #'normalize__with_std': ['None' if (elem['pipeline']['normalize'][0] == 'normalize_NoneType' or elem['pipeline']['normalize'][0] == 'normalize_MinMaxScaler') else elem['pipeline']['normalize'][1]['normalize__with_std']], 
                'outlier': ['None' if elem['pipeline']['outlier'][0] == 'outlier_NoneType' else elem['pipeline']['outlier'][0]], 
                'outlier__n_neighbors': ['None' if elem['pipeline']['outlier'][0] == 'outlier_NoneType' or elem['pipeline']['outlier'][0] =='outlier_IsolationOutlierDetector' else elem['pipeline']['outlier'][1]['outlier__n_neighbors']], 
                'algorithm': [elem['algorithm'][0]], 
                #'algorithm__max_iter': [elem['algorithm'][1]['max_iter']], 
                'algorithm__n_clusters': [elem['algorithm'][1]['n_clusters']], 
                'optimization_internal_metric': [metric[0]], 
                'optimization_external_metric': ['ami'], 
                'optimization_internal_metric_value': [elem['internal_metric']], 
                'optimization_external_metric_value': [elem['external_metric']],
                'max_optimization_internal_metric_value': [elem['max_history_internal_metric']], 
                'max_optimization_external_metric_value': [elem['max_history_external_metric']],
                'duration': [elem['duration']],
                'budget': [metric[1]],
                'tot_conf': tot_conf
            }), ignore_index=True)

    return results

def save_result(results, output_path, dataset, metric):
    results.to_csv(os.path.join(output_path, dataset + '_' + metric + '_results.csv'), index=False)