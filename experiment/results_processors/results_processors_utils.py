import os
import json
import argparse
import openml

import pandas as pd

from os import listdir
from os.path import isfile, join
from matplotlib import gridspec


def get_dataset(name):
    data = pd.read_csv(os.path.join("datasets", name + ".csv"))
    PrototypeSingleton.getInstance().setDatasetIndex(data["index"].to_numpy())
    data = data.drop(["index"], axis=1)
    feature_names = [column for column in data.columns if column != "target"]
    data = data.to_numpy()
    X, y = data[:, :-1], data[:, -1]
    categorical_indicator = [False for _ in range(X.shape[1])]
    set_singleton(name, X, y, categorical_indicator, feature_names)
    return X, y, feature_names


def set_singleton(dataset_name, X, y, categorical_indicator, dataset_features_names):
    num_features = [i for i, x in enumerate(categorical_indicator) if x == False]
    cat_features = [i for i, x in enumerate(categorical_indicator) if x == True]
    PrototypeSingleton.getInstance().setFeatures(num_features, cat_features)
    PrototypeSingleton.getInstance().set_X_y(X, y)
    PrototypeSingleton.getInstance().setDatasetFeaturesName(dataset_features_names)


def load_result(input_path, dataset, info):
    results = pd.DataFrame()
    file_name = info["file_name"] + ".json"
    _, _, original_features = get_dataset(dataset)
    num_features = len(original_features)
    tot_conf = (
        (44 * num_features)
        if info["space"] == "toy"
        else (2310 * (1 + 4 * (num_features - 1)))
    )
    with open(os.path.join(input_path, file_name)) as json_file:
        data = json.load(json_file)
        history = data["context"]["history"]
        for elem in history:
            results = results.append(
                pd.DataFrame(
                    {
                        "dataset": [dataset],
                        "iteration": [elem["iteration"]],
                        #'pipeline': [elem['pipeline']],
                        #'algorithm': [elem['algorithm']],
                        "features": [
                            "None"
                            if elem["pipeline"]["features"][0] == "features_NoneType"
                            else elem["pipeline"]["features"][0]
                        ],
                        "features__k": [
                            "None"
                            if (
                                elem["pipeline"]["features"][0] == "features_NoneType"
                                or elem["pipeline"]["features"][0]
                                == "features_PearsonThreshold"
                            )
                            else elem["pipeline"]["features"][1]["features__k"]
                        ],
                        "features__threshold": [
                            "None"
                            if elem["pipeline"]["features"][0]
                            != "features_PearsonThreshold"
                            else elem["pipeline"]["features"][1]["features__threshold"]
                        ],
                        "normalize": [
                            "None"
                            if elem["pipeline"]["normalize"][0] == "normalize_NoneType"
                            else elem["pipeline"]["normalize"][0]
                        ],
                        #'normalize__with_mean': ['None' if (elem['pipeline']['normalize'][0] == 'normalize_NoneType' or elem['pipeline']['normalize'][0] == 'normalize_MinMaxScaler') else elem['pipeline']['normalize'][1]['normalize__with_mean']],
                        #'normalize__with_std': ['None' if (elem['pipeline']['normalize'][0] == 'normalize_NoneType' or elem['pipeline']['normalize'][0] == 'normalize_MinMaxScaler') else elem['pipeline']['normalize'][1]['normalize__with_std']],
                        "outlier": [
                            "None"
                            if elem["pipeline"]["outlier"][0] == "outlier_NoneType"
                            else elem["pipeline"]["outlier"][0]
                        ],
                        "outlier__n_neighbors": [
                            "None"
                            if elem["pipeline"]["outlier"][0] == "outlier_NoneType"
                            or elem["pipeline"]["outlier"][0]
                            == "outlier_IsolationOutlierDetector"
                            else elem["pipeline"]["outlier"][1]["outlier__n_neighbors"]
                        ],
                        "algorithm": [elem["algorithm"][0]],
                        #'algorithm__max_iter': [elem['algorithm'][1]['max_iter']],
                        "algorithm__n_clusters": [elem["algorithm"][1]["n_clusters"]],
                        "optimization_internal_metric": [
                            info["optimization_internal_metric"]
                        ],
                        "optimization_external_metric": ["ami"],
                        "optimization_internal_metric_value": [elem["internal_metric"]],
                        "optimization_external_metric_value": [elem["external_metric"]],
                        "max_optimization_internal_metric_value": [
                            elem["max_history_internal_metric"]
                        ],
                        "max_optimization_external_metric_value": [
                            elem["max_history_external_metric"]
                        ],
                        "duration": [elem["duration"]],
                        "budget": [info["budget"]],
                        "tot_conf": tot_conf,
                    }
                ),
                ignore_index=True,
            )

    return results


def save_result(results, output_path, dataset, metric):
    results.to_csv(
        os.path.join(output_path, dataset + "_" + metric + "_results.csv"), index=False
    )


def join_indices(dfs):
    return [
        row["index"]
        for _, row in dfs[0].iterrows()
        if row["index"] in list(dfs[1]["index"])
    ]


def drop_index_col_from(df):
    return [column for column in list(df.columns) if column != "index"]


def filter_dfs(dfs, return_type):

    indices = join_indices(dfs)
    # print("#################### INDICES ####################")
    # print(pd.DataFrame(indices))

    to_return = [
        return_type(
            [
                row[drop_index_col_from(df)]
                for _, row in df.iterrows()
                if row["index"] in indices
            ]
        )
        for df in dfs
    ]
    # print(pd.DataFrame(to_return[0]))
    # print(pd.DataFrame(to_return[1]))
    # print("\n\n\n\n")
    return to_return
