# Author: Lars Buitinck
# License: 3-clause BSD
from numbers import Real

import numpy as np
import pandas as pd


from fsfc.base import BaseFeatureSelector
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class PearsonThreshold(BaseFeatureSelector):
    """Feature selector that removes all correlated features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Read more in the :ref:`User Guide <variance_threshold>`.

    Parameters
    ----------
    threshold : float, default=0
        Maximum correlation allowed, remove the features that have the same value or greater.

    """

    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.features = None
        self.original_features = None
        self.to_drops = None

    def fit(self, X, y=None):
        """Compability for supervised techniques.
        """
        df = pd.DataFrame(X)
        corr = abs(df.corr())
        self.features = range(X.shape[-1])
        self.original_features = range(X.shape[-1])
        self.to_drops = []
        for i in range(X.shape[-1]):
            if i in self.features:
                to_drop = [idx for idx, elem in enumerate(corr.iloc[:, i] > self.threshold) if elem and idx != i]
                self.to_drops.append((i, to_drop))
        feature_sets = self.find_maximum_feature_sets(self.to_drops)
        # print(f"CANDIDATES:\t{features}")
        results = {}
        for feature_set in feature_sets:
            # print(f"\tcandidate\t{feature_set}")
            Xt = X[:, feature_set]
            results[str(feature_set)] = -1
            for k in range(2, 13):
                current_sil = silhouette_score(Xt, KMeans(n_clusters=k, random_state=0).fit_predict(Xt))
                results[str(feature_set)] = max(results[str(feature_set)], current_sil)
                # print(f"\t\tk={k}, silhouette:{current_sil}")
        self.features = np.fromstring(max(results, key=results.get)[1:-1], dtype=int, sep=',')
        return self

    def _get_support_mask(self):
        return [elem in self.features for elem in self.original_features]

    def transform(self, X, y=None):
        return X[:, self.features]

    def is_valid_set(self, feature_set, z):
        for x, y_list in z:
            if x in feature_set:
                for y in y_list:
                    if y in feature_set:
                        return False
        return True

    def backtrack(self, z, current_set, index, result):
        if index == len(z):
            result.append(current_set[:])
            return

        for i in range(index, len(z)):
            x, _ = z[i]
            if x not in current_set and self.is_valid_set(current_set + [x], z):
                current_set.append(x)
                self.backtrack(z, current_set, i + 1, result)
                current_set.pop()

    def find_maximum_feature_sets(self, z):
        result = []
        self.backtrack(z, [], 0, result)

        max_length = max(len(feature_set) for feature_set in result)
        max_feature_sets = [feature_set for feature_set in result if len(feature_set) == max_length]
        return max_feature_sets
