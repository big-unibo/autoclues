import numpy as np

from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestCentroid
import math

def my_silhouette_samples(X, labels, *, metric="euclidean", **kwds):
    """Compute the Silhouette Coefficient for each sample.
    The Silhouette Coefficient is a measure of how well samples are clustered
    with samples that are similar to themselves. Clustering models with a high
    Silhouette Coefficient are said to be dense, where samples in the same
    cluster are similar to each other, and well separated, where samples in
    different clusters are not very similar to each other.
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.
    Note that Silhouette Coefficient is only defined if number of labels
    is 2 ``<= n_labels <= n_samples - 1``.
    This function returns the Silhouette Coefficient for each sample.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters.
    Read more in the :ref:`User Guide <silhouette_coefficient>`.
    Parameters
    ----------
    X : array-like of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.
    labels : array-like of shape (n_samples,)
        Label values for each sample.
    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.
    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    silhouette : array-like of shape (n_samples,)
        Silhouette Coefficients for each sample.
    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
       <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
    """
    import functools
    from sklearn.utils import check_random_state
    from sklearn.utils import check_X_y
    from sklearn.utils import _safe_indexing
    from sklearn.metrics.pairwise import pairwise_distances_chunked
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.preprocessing import LabelEncoder
    X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])

    def check_number_of_labels(n_labels, n_samples):
        """Check that number of labels are valid.
        Parameters
        ----------
        n_labels : int
            Number of labels.
        n_samples : int
            Number of samples.
        """
        if not 1 < n_labels < n_samples:
            raise ValueError(
                "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
                % n_labels
            )

    def _silhouette_reduce(D_chunk, start, labels, label_freqs):
        """Accumulate silhouette statistics for vertical chunk of X.
        Parameters
        ----------
        D_chunk : array-like of shape (n_chunk_samples, n_samples)
            Precomputed distances for a chunk.
        start : int
            First index in the chunk.
        labels : array-like of shape (n_samples,)
            Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
        label_freqs : array-like
            Distribution of cluster labels in ``labels``.
        """
        # accumulate distances from each sample to each cluster
        clust_dists = np.zeros((len(D_chunk), len(label_freqs)), dtype=D_chunk.dtype)
        for i in range(len(D_chunk)):
            clust_dists[i] += np.bincount(
                labels, weights=D_chunk[i], minlength=len(label_freqs)
            )

        # intra_index selects intra-cluster distances within clust_dists
        intra_index = (np.arange(len(D_chunk)), labels[start : start + len(D_chunk)])
        # intra_clust_dists are averaged over cluster size outside this function
        intra_clust_dists = clust_dists[intra_index]
        # of the remaining distances we normalise and extract the minimum
        clust_dists[intra_index] = np.inf
        clust_dists /= label_freqs
        inter_clust_dists = clust_dists.min(axis=1)
        return intra_clust_dists, inter_clust_dists

    # Check for non-zero diagonal entries in precomputed distance matrix
    if metric == "precomputed":
        atol = np.finfo(X.dtype).eps * 100
        if np.any(np.abs(np.diagonal(X)) > atol):
            raise ValueError(
                "The precomputed distance matrix contains non-zero "
                "elements on the diagonal. Use np.fill_diagonal(X, 0)."
            )

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    check_number_of_labels(len(le.classes_), n_samples)

    kwds["metric"] = metric
    reduce_func = functools.partial(
        _silhouette_reduce, labels=labels, label_freqs=label_freqs
    )
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)

    denom = (label_freqs - 1).take(labels, mode="clip")
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    sil_samples = inter_clust_dists - intra_clust_dists
    with np.errstate(divide="ignore", invalid="ignore"):
        sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples), np.nan_to_num(intra_clust_dists), np.nan_to_num(inter_clust_dists)

def get_shapes(X, Xt):
    return X.shape[1], Xt.shape[1]

def weighted_metric(X, Xt, labels, mode, indeces):
    method, weighting = mode.split("-")
    if method == "lensen":
        return -1 * weighted_Lensen(X, Xt, labels, weighting, indeces)
    elif method == "hancer":
        if weighting == "extended":
            return weighted_Hancer_extended(X, Xt, labels)
        else:
            return weighted_Hancer(X, Xt, labels, weighting, indeces)
    else:
        raise Exception("Objective not valid")

def weighted_Lensen(X, Xt, labels, mode, indeces):
    # This has to be maximized
    # All features are considered in the calculation of the silhouette (actually in the paper they use inter/intra) and then a weightening function is applied
    # We use the inter/intra
    if mode not in ["linear", "nonlinear"]:
        raise Exception("Objective not valid")

    if indeces:
        X = X[indeces]
    all_features, selected_features = get_shapes(X, Xt)
    _, intra_clust_dists, _ = my_silhouette_samples(X, labels)
    intra_final = np.power(intra_clust_dists, 2).sum()/X.shape[0]

    centroid_calculator = NearestCentroid().fit(X, labels)
    centroids = centroid_calculator.centroids_
    classes = centroid_calculator.classes_
    mean_centroid = NearestCentroid().fit(np.concatenate((centroids, np.ones((1, X.shape[1]))), axis=0), np.concatenate((np.zeros(centroids.shape[0]), np.ones(1)), axis=0)).centroids_[0]
    inter_final = sum([np.power(np.linalg.norm(centroid - mean_centroid), 2) * len(np.where(labels == idx))/len(labels) for idx, centroid in zip(classes, centroids)])

    final_metric = inter_final/intra_final

    feature_weighting = (
        feature_weighting_Lensen_linear(all_features, selected_features)
        if mode == "linear" else
        feature_weighting_Lensen_non_linear(all_features, selected_features)
        )
    return final_metric * feature_weighting

def feature_weighting_Lensen_linear(all_features, selected_features):
    # It privileges lower dimensions because the function (https://www.wolframalpha.com/input?i=y+%3D+%2810-x%29%2F10) has to be maximized
    value = (all_features - selected_features) / all_features
    max_value = (all_features - 1) / all_features
    return value / max_value

def feature_weighting_Lensen_non_linear(all_features, selected_features):
    # It privileges lower dimensions because the function (https://www.wolframalpha.com/input?i=1%2F10+*+sqrt%2810%5E2+-+x%5E2%29) has to be maximized
    return (1/(all_features+1)) * math.sqrt(math.pow(all_features + 1, 2) - math.pow(selected_features, 2))


def weighted_Hancer(X, Xt, labels, mode, indeces):
    # This has to be minimized
    # All features are considered in the calculation of the silhouette (actually in the paper they use intra/inter) and then a weightening function is applied
    # We use 1/silhouette
    if mode not in ["linear", "nonlinear"]:
        raise Exception("Objective not valid")
    if indeces:
        X = X[indeces]
    all_feature_silhouette = silhouette_score(X, labels)
    all_features, selected_features = get_shapes(X, Xt)
    feature_weighting = (
        feature_weighting_Hancer_linear(all_features, selected_features)
        if mode == "linear" else
        feature_weighting_Hancer_non_linear(all_features, selected_features)
        )
    return (1/all_feature_silhouette) * feature_weighting

def feature_weighting_Hancer_linear(all_features, selected_features):
    # It privileges lower dimensions because the function (https://www.wolframalpha.com/input?i=x%2F%2810-x%29) has to be minimized
    try:
        max_value = (all_features - 1) / (all_features - selected_features)
        value = selected_features / (all_features - selected_features)
        return value / max_value
    except:
        return 1

def feature_weighting_Hancer_non_linear(all_features, selected_features):
    # It discourages dimensions near the mean because the function (https://www.wolframalpha.com/input?i=e%5E%28%28%28x-5%29%2F%282*5.3%5E2%29%29%5E-1%29) has to be minimized
    mu = all_features / 2
    var = math.pow(all_features - 2, 2) / 12
    max_value = math.pow(math.e, math.pow(1/(2 * var), -1))
    min_value = 0
    try:
        value = math.pow(math.e, math.pow((selected_features - mu)/(2 * var), -1))
        return value / max_value
    except:
        return min_value


def weighted_Hancer_extended(X, Xt, labels):
    # This has to be minimized
    # Only the selected features are considered in the calculation of the silhouette (actually in the paper they use 1/sil) and then a weightening function is applied
    # We use 1/silhouette
    selected_feature_silhouette = silhouette_score(Xt, labels)
    if selected_feature_silhouette < 0:
        raise Exception("Negative Silhouette")
    feature_weighting = feature_weighting_Hancer_extended(X.shape[1], Xt.shape[1])
    return (1/selected_feature_silhouette) * feature_weighting

def feature_weighting_Hancer_extended(all_features, selected_features):
    # It privileges higher dimensions because the function (https://www.wolframalpha.com/input?i=%2810-x%29%2F%2810-1%29) has to be minimized
    return (all_features + 1 - selected_features) / (all_features + 1 - 1)

