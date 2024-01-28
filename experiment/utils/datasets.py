import pandas as pd
import numpy as np
import os
import openml

from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    load_digits,
    fetch_covtype,
)
from pipeline.PrototypeSingleton import PrototypeSingleton


# def get_dataset(name):
#     loader = {
#         "blood": 1464,
#         "breast": 1465,  # this is breast-tissue, not breast cancer
#         "diabetes": 37,
#         "ecoli": 40671,
#         "iris": 61,
#         "parkinsons": 1488,
#         "seeds": 1499,
#         "thyroid": 40682,
#         "vehicle": 54,
#         "wine": 187,
#     }
#     if name in loader:
#         return load_dataset_from_openml(loader[name])
#     else:
#         return load_dataset_from_csv(name)


# def load_dataset_from_openml(id):
#     dataset = openml.datasets.get_dataset(id)
#     X, y, categorical_indicator, _ = dataset.get_data(
#         dataset_format="array", target=dataset.default_target_attribute
#     )
#     dataset_features_names = [str(elem) for elem in list(dataset.features.values())]
#     dataset_features_names = dataset_features_names[:-1]
#     set_singleton(dataset.name, X, y, categorical_indicator, dataset_features_names)
#     return X, y, dataset_features_names


# def breast_cancer():
#     data = load_breast_cancer()
#     return data.data, data.target, data.feature_names


# def iris():
#     data = load_iris()
#     return data.data, data.target, data.feature_names


# def wine():
#     data = load_wine()
#     return data.data, data.target, data.feature_names


# def digits():
#     digits = load_digits()
#     n_samples = len(digits.images)
#     data = digits.images.reshape((n_samples, -1))
#     return data, digits.target, data.feature_names


# def covtype():
#     data = fetch_covtype()
#     return data.data, data.target, data.feature_names


# def load_dataset_from_csv(name):
#     data = pd.read_csv(os.path.join("datasets", name + ".csv"), header=None)
#     data = data.to_numpy()
#     if name == "parkinsons":
#         features_name = [
#             "MDVP:Fo(Hz)",
#             "MDVP:Fhi(Hz)",
#             "MDVP:Flo(Hz)",
#             "MDVP:Jitter(%)",
#             "MDVP:Jitter(Abs)",
#             "MDVP:RAP",
#             "MDVP:PPQ",
#             "Jitter:DDP",
#             "MDVP:Shimmer",
#             "MDVP:Shimmer(dB)",
#             "Shimmer:APQ3",
#             "Shimmer:APQ5",
#             "MDVP:APQ",
#             "Shimmer:DDA",
#             "NHR",
#             "HNR",
#             "RPDE",
#             "DFA",
#             "spread1",
#             "spread2",
#             "D2",
#             "PPE",
#         ]
#     elif name == "seeds":
#         features_name = [
#             "area",
#             "perimeter",
#             "compactness",
#             "length of kernel",
#             "width of kernel",
#             "asymmetry coefficient",
#             "length of kernel groove",
#         ]
#     elif name == "synthetic":
#         features_name = [
#             "feature_0",
#             "feature_1",
#             "feature_2",
#             "feature_3",
#             "feature_4",
#         ]
#     elif name == "iris2" or name == "iris3":
#         features_name = [
#             "[2 - petallength (numeric)]",
#             "[3 - petalwidth (numeric)]",
#         ]
#     else:
#         features_name = [f"feature_{idx}" for idx in range(data.shape[1] - 1)]
#         # raise Exception('No features names assigned')
#     X, y = data[:, :-1], data[:, -1]
#     categorical_indicator = [False for _ in range(X.shape[1])]
#     set_singleton(name, X, y, categorical_indicator, features_name)
#     return X, y, features_name


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
    # print("DATASET:")
    # print("#" * 50)
    # print(f"\tname:\t{dataset_name}")
    # print(f"\tshape:\t{X.shape}")
    # print(
    #     f"\tnumerical features:\t{len(num_features)}\n\tcategorical features:\t{len(cat_features)}"
    # )
    # print(f"\tfirst instance of X:\t{X[0, :]}")
    # print(f"\tfirst instance of y:\t{y[0]}")
    # print("#" * 50 + "\n")
