import os

# from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour
# from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    KBinsDiscretizer,
    Binarizer,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.neighbors import LocalOutlierFactor

from .utils import generate_domain_space

from pipeline.outlier_detectors import (
    LocalOutlierDetector,
    IsolationOutlierDetector,
)  # , SGDOutlierDetector
from pipeline.feature_selectors import PearsonThreshold
from pipeline import space
from fsfc.generic import GenericSPEC, NormalizedCut, WKMeans
import pandas as pd
import numpy as np


class PrototypeSingleton:
    __instance = None

    POOL = {
        "normalize": [
            None,
            StandardScaler(),
            MinMaxScaler(),
            RobustScaler(),
            PowerTransformer(),
        ],
        "outlier": [None, LocalOutlierDetector(), IsolationOutlierDetector()],
        "features": [
            None,
            PearsonThreshold(threshold=0.5),
            GenericSPEC(k=2),
            WKMeans(k=2, beta=0),
        ],
    }

    features_names = []
    PROTOTYPE = {}
    DOMAIN_SPACE = {}
    parts = []
    X = []
    y = []
    index = []
    original_numerical_features = []
    original_categorical_features = []
    current_numerical_features = []
    current_categorical_features = []

    @staticmethod
    def getInstance():
        """Static access method."""
        if PrototypeSingleton.__instance == None:
            PrototypeSingleton()
        return PrototypeSingleton.__instance

    def __init__(self):
        """Virtually private constructor."""
        if PrototypeSingleton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            PrototypeSingleton.__instance = self

    def setDatasetFeaturesName(self, dataset_features_names):
        self.features_names = dataset_features_names
        space.num_features = len(dataset_features_names)

    def getFeaturesFromMask(self, mask=[], add_Nones=False):
        if mask == []:
            names = self.features_names
        else:
            indices = [i for i, x in enumerate(mask) if x]
            names = [self.features_names[i] for i in indices]

        while len(names) < 3:
            names.append("None")
        return names

    def setPipeline(self, params, space):
        for param in params:
            self.parts.append(param)

        for part in self.parts:
            self.PROTOTYPE[part] = [
                elem
                for elem in self.POOL[part]
                if (elem == None) or (elem.__class__.__name__ in space.keys())
            ]

        self.DOMAIN_SPACE = generate_domain_space(self.PROTOTYPE, space)

    def setFeatures(self, num_features, cat_features):
        self.original_numerical_features = num_features
        self.current_numerical_features = num_features
        self.original_categorical_features = cat_features
        self.current_categorical_features = cat_features

    def set_X_y(self, X, y):
        self.X = X
        self.y = y

    def setDatasetIndex(self, index):
        self.index = index.copy()

    def resetFeatures(self):
        self.current_numerical_features = self.original_numerical_features
        self.current_categorical_features = []
        self.current_categorical_features.extend(self.original_categorical_features)

    def applyColumnTransformer(self):
        len_numerical_features = len(self.current_numerical_features)
        len_categorical_features = len(self.current_categorical_features)
        self.current_numerical_features = list(range(0, len_numerical_features))
        self.current_categorical_features = list(
            range(
                len_numerical_features,
                len_categorical_features + len_numerical_features,
            )
        )

    def applyFeaturesEngineering(self, indeces):
        # print(indeces)
        # self.current_numerical_features = [i for i in self.current_numerical_features if indeces[i] == True]
        # self.current_categorical_features = [i for i in self.current_categorical_features if indeces[i] == True]
        new_i = 0
        new_numerical_features, new_categorical_features = [], []
        for i in range(len(indeces)):
            if indeces[i] == True:
                if i in self.current_numerical_features:
                    new_numerical_features += [new_i]
                else:  # if i in self.current_categorical_features:
                    new_categorical_features += [new_i]
                new_i += 1
        self.current_numerical_features = new_numerical_features
        self.current_categorical_features = new_categorical_features

    def getFeatures(self):
        return self.current_numerical_features, self.current_categorical_features

    def getDomainSpace(self):
        return self.DOMAIN_SPACE

    def getPrototype(self):
        return self.PROTOTYPE

    def getParts(self):
        return self.parts

    def getX(self):
        return self.X
