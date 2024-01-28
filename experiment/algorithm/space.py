from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    Birch,
)
from sklearn.mixture import GaussianMixture

from hyperopt import hp

algorithms = {
    "KMeans": KMeans,
    # Same of KMeans but more efficient
    "MiniBatchKMeans": MiniBatchKMeans,
    "AffinityPropagation": AffinityPropagation,
    "MeanShift": MeanShift,
    "SpectralClustering": SpectralClustering,
    "AgglomerativeClustering": AgglomerativeClustering,
    # It gives overlapping clusters
    "GaussianMixture": GaussianMixture,
    # We do not know the search space
    "OPTICS": OPTICS,
    "Birch": Birch,
    # We have import problems because of the legacy version of this tool
    # 'KMedoids': KMedoids,
}

grid_k_means = {"n_clusters": list(range(2, 201))}

grid_mini_batch_k_means = {"n_clusters": list(range(2, 201))}

grid_affinity_propagation = {}

grid_mean_shift = {
    # "bandwidth": list(range(2, 201))
}

grid_spectral_clustering = {"n_clusters": list(range(2, 201))}

grid_agglomerative_clustering = {
    "n_clusters": list(range(2, 201)),
    "linkage": ["complete"],
}

grid_gaussian_mixture = {"n_components": list(range(2, 201))}

# I do not know the right search space
grid_optics = {"min_samples": list(range(2, 201))}

grid_birch = {"n_clusters": list(range(2, 201))}

# grid_k_medoids = {
#     "n_clusters": list(range(2, 201))
# }

parameter_grid = {
    "KMeans": grid_k_means,
    "MiniBatchKMeans": grid_mini_batch_k_means,
    "AffinityPropagation": grid_affinity_propagation,
    "MeanShift": grid_mean_shift,
    "SpectralClustering": grid_spectral_clustering,
    "AgglomerativeClustering": grid_agglomerative_clustering,
    "GaussianMixture": grid_gaussian_mixture,
    "OPTICS": grid_optics,
    "Birch": grid_birch,
    # 'KMedoids': grid_k_medoids,
}


def get_domain_space(input_pace, max_k):
    algorithms_space = []
    print("\tclustering")
    param_grid = {
        algorithm: grid
        for algorithm, grid in parameter_grid.items()
        if algorithm in input_pace.keys()
    }
    for algorithm, grid in param_grid.items():
        print(f"""\t\t{algorithm}""")
        algorithm_config = {}
        to_print_algorithm_config = ""
        for hyperparameter, domain in grid.items():
            if (
                hyperparameter == "n_clusters"
                or hyperparameter == "n_components"
                or hyperparameter == "bandwidth"
            ):
                domain = list(range(2, max_k + 1))
            if hyperparameter in input_pace[algorithm]:
                domain = input_pace[algorithm][hyperparameter]
            to_print_algorithm_config += "\t\t\t{}: {}\n".format(hyperparameter, domain)
            algorithm_config[hyperparameter] = hp.choice(
                "{}_{}".format(algorithm, hyperparameter), domain
            )
        algorithms_space.append((algorithm, algorithm_config))
        print(to_print_algorithm_config)
    print("#" * 50 + "\n")
    return hp.choice("algorithm", algorithms_space)
