import json
import os
import random

from matplotlib import cm
from results_processors.results_processors_utils import get_dataset

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from pipeline.feature_selectors import PearsonThreshold
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


# from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour
# from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
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
    FunctionTransformer,
)
from pipeline.feature_selectors import PearsonThreshold

from imblearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion

from pipeline.PrototypeSingleton import PrototypeSingleton
from pipeline.outlier_detectors import (
    LocalOutlierDetector,
    IsolationOutlierDetector,
)  # , SGDOutlierDetector

from fsfc.generic import GenericSPEC, NormalizedCut, WKMeans

from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

import matplotlib.pyplot as plt
import plotly.express as px




def single_plot(ax, df, target_column, unique_clusters, type, colors):
    if type != "PARA":
        if df.shape[1] > 3:
            Xt = pd.concat(
                [
                    pd.DataFrame(
                        TSNE(n_components=2, random_state=42).fit_transform(
                            # pd.read_csv(
                            #     "results/optimization/smbo/details/ecoli_sil/ecoli_sil_478_X_normalize.csv"
                            # ).to_numpy(),
                            df.iloc[:, :-1].to_numpy(),
                            df.iloc[:, -1].to_numpy(),
                        )
                        if type == "TSNE"
                        else PCA(n_components=2, random_state=42).fit_transform(
                            # pd.read_csv(
                            #     "results/optimization/smbo/details/ecoli_sil/ecoli_sil_478_X_normalize.csv"
                            # ).to_numpy(),
                            df.iloc[:, :-1].to_numpy(),
                            df.iloc[:, -1].to_numpy(),
                        ),
                        columns=[f"{type}_0", f"{type}_1"],
                    ),
                    df[target_column],
                ],
                axis=1,
            )
        else:
            Xt = df
        # print(Xt)

        for i, cluster_label in enumerate(unique_clusters):
            cluster_data = Xt[Xt[target_column] == cluster_label]
            plt.scatter(
                cluster_data.iloc[:, 0],
                cluster_data.iloc[:, 1],
                c=[colors[i]] * cluster_data.shape[0],
                label=f"Cluster {cluster_label}",
            )

            n_selected_features = Xt.shape[1]
            Xt = Xt.iloc[:, :n_selected_features]
            min, max = Xt.min().min(), Xt.max().max()
            range = (max - min) / 10
            ax.set_xlim([min - range, max + range])
            ax.set_ylim([min - range, max + range])
            ax.set_xlabel("TSNE feat.", fontsize=35)
            ax.set_ylabel("TSNE feat.", fontsize=35)
            ax.yaxis.labelpad = 10
            ax.xaxis.labelpad = 10
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.tick_params(axis='both', which='minor', labelsize=28)
    else:
        ax = pd.plotting.parallel_coordinates(df, "target", color=colors)
    # ax.set_title(type)


def plot_cluster_data(df, target_column, colors, returned_types="all"):
    """
    Plot a dataframe with clustering results using a scatter plot.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to be plotted.
    target_column (str): The name of the column that contains the cluster labels.

    Returns:
    None
    """
    subplot_types = ["TSNE", "PCA", "PARA"] if returned_types == "all" else returned_types

    # Make sure the target_column is in the dataframe
    if target_column not in df.columns:
        raise ValueError(f"'{target_column}' column not found in the dataframe.")

    # Create a scatter plot for each cluster
    unique_clusters = df[target_column].unique()

    fig = plt.figure(figsize=(7 * len(subplot_types), 5))
    for idx, subplot_type in enumerate(subplot_types):
        ax = fig.add_subplot(1, len(subplot_types), idx + 1)
        single_plot(
            ax=ax,
            df=df,
            target_column=target_column,
            unique_clusters=unique_clusters,
            type=subplot_type,
            colors=colors
        )

    return fig


if __name__ == "__main__":

    random.seed(42)
    np.random.seed(42)

    output_path = os.path.join("/", "home", "toy_dashboard")

    result = {}
    for comb in [
        "cl",
        "ft_cl",
        "ft_sc_cl",
        "ft_sc_ou_cl"
        ]:


        result[comb] = {}
        max_dict = {
            "internal_metric": -2,
            "external_metric": 0,
            "n_clusters": 1,
        }
        print("pipeline\tn_clusters\tsilhouette\tsilhouette_TSNE\t\tami")
        for n_clusters in range(2, 15):

            steps = comb.split("_")

            X, y, dataset_features_names = get_dataset("syn2")

            bug_avoided = False
            while bug_avoided == False:
                try:
                    pipe_array = [
                        ("features", WKMeans(k=5, beta=-7) if "ft" in steps else FunctionTransformer()),
                        ("scaler", MinMaxScaler() if "sc" in steps else FunctionTransformer()),
                        ("outlier", IsolationOutlierDetector(n_estimators=100,random_state=42) if "ou" in steps else FunctionTransformer()),
                        ("clustering", AgglomerativeClustering(n_clusters=n_clusters, linkage="complete")),
                    ]
                    pipe = Pipeline(pipe_array)

                    if len(steps) > 1:
                        if "ou" in steps:
                            Xt, yt = pipe[:-1].fit_resample(X.copy(), y.copy())
                        else:
                            Xt = pipe[:-1].fit_transform(X.copy())
                            yt = y
                        columns = [feature for idx, feature in enumerate(dataset_features_names) if pipe[0]._get_support_mask()[idx]]
                    else:
                        Xt, yt = X, y
                        columns = dataset_features_names
                    y_pred = pipe.fit_predict(X.copy(), y.copy())
                    # print(y_pred.shape)
                    # print(Xt.shape)

                    internal_metric_plain = round(silhouette_score(
                            Xt,
                            y_pred),
                            2)
                    internal_metric_TSNE = round(silhouette_score(
                        TSNE(n_components=2, random_state=42).fit_transform(
                            Xt,
                            y_pred),
                            y_pred),
                            2)
                    external_metric = round(adjusted_mutual_info_score(yt, y_pred), 2)
                    bug_avoided = True
                except:
                    pass

                if bug_avoided == True:
                    result[comb][n_clusters] = {
                        "sil": internal_metric_plain,
                        "sil_TSNE": internal_metric_TSNE,
                        "ami": external_metric,
                        "Xt": Xt,
                        "yt": yt,
                        "y_pred": y_pred,
                    }
                    internal_metric = internal_metric_plain if len(steps) == 1 else internal_metric_TSNE
                    if internal_metric > max_dict["internal_metric"]:
                        max_dict["internal_metric"] = internal_metric
                        max_dict["external_metric"] = external_metric
                        max_dict["n_clusters"] = n_clusters

            print(f"{comb}\t\t{n_clusters}\t\t{internal_metric_plain}\t\t{internal_metric_TSNE}\t\t{external_metric}")
        print()

        pd.DataFrame(result[comb]).T[["sil", "ami", "sil_TSNE"]].to_csv(os.path.join(output_path, f"{comb}.csv"))
        df = pd.concat(
            [
                pd.DataFrame(result[comb][max_dict["n_clusters"]]["Xt"], columns=columns),
                pd.DataFrame(result[comb][max_dict["n_clusters"]]["y_pred"], columns=["target"])
            ],
            axis=1)

        colors = cm.rainbow(np.linspace(0, 1, len(np.unique(result[comb][max_dict["n_clusters"]]["y_pred"]))))
        fig = plot_cluster_data(df, "target", colors, ["TSNE"])

        # Save the figure as a PNG file
        fig.savefig(os.path.join(output_path, f"{comb}.pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(output_path, f"{comb}.png"), dpi=300, bbox_inches="tight")

        # Close the figure (optional)
        plt.close(fig)
