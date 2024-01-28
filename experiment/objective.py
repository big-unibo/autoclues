import hashlib
import json
import time
import sys
import os
import traceback


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from hyperopt import STATUS_OK, STATUS_FAIL
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from s_dbw import S_Dbw
from sklearn import metrics

from algorithm import space as ALGORITHM_SPACE
from pipeline.prototype import pipeline_conf_to_full_pipeline
from utils.metrics import my_silhouette_samples, weighted_metric
from pipeline.PrototypeSingleton import PrototypeSingleton


def objective(pipeline_config, algo_config, X, y, context, config):
    pipeline_hash = hashlib.sha1(
        json.dumps(pipeline_config, sort_keys=True).encode()
    ).hexdigest()
    algorithm_hash = hashlib.sha1(
        json.dumps(algo_config, sort_keys=True).encode()
    ).hexdigest()
    item_hash = {
        "pipeline": pipeline_hash,
        "algorithm": algorithm_hash,
        "config": hashlib.sha1(
            str(pipeline_hash + algorithm_hash).encode()
        ).hexdigest(),
    }

    item = {
        "pipeline": pipeline_config,
        "algorithm": algo_config,
    }

    algorithm = algo_config[0]
    algo_config = algo_config[1]

    try:
        pipeline, _ = pipeline_conf_to_full_pipeline(
            pipeline_config,
            ALGORITHM_SPACE.algorithms.get(algorithm),
            config["seed"],
            algo_config,
        )
    except Exception as e:
        print(e)

    # print(pipeline)
    history_index = context["history_index"].get(item_hash["config"])
    if history_index is not None:
        return context["history"][history_index]

    start = time.time()
    try:
        Xt, yt = X.copy(), y.copy().reshape(-1, 1).astype(int)
        Xt_to_export, yt_to_export = {}, {}
        labels = PrototypeSingleton.getInstance().getFeaturesFromMask()

        indeces = (
            PrototypeSingleton.getInstance().index.copy().reshape(-1, 1).astype(int)
        )
        Xt_to_export["original"] = pd.DataFrame(
            np.concatenate([indeces, Xt.copy()], axis=1),
            columns=["index"] + [l for l in labels if l != "None"],
        )
        yt_to_export["original"] = pd.DataFrame(
            np.concatenate([indeces, yt.copy()], axis=1), columns=["index", "target"]
        )
        if len(pipeline.steps) > 1:
            for step, operator in pipeline[:-1].named_steps.items():
                if step == "outlier":
                    Xt, yt = operator.fit_resample(Xt, yt)
                    yt = yt.reshape(-1, 1).astype(int)
                    indeces = operator.indeces.reshape(-1, 1).astype(int)
                    yt_to_export[step] = pd.DataFrame(
                        np.concatenate([indeces, yt.copy()], axis=1),
                        columns=["index", "target"],
                    )
                    # PrototypeSingleton.getInstance().setDatasetIndex(indeces)
                else:
                    Xt = operator.fit_transform(Xt, None)
                    if step == "features":
                        mask = pipeline["features"].get_support()
                        labels = PrototypeSingleton.getInstance().getFeaturesFromMask(
                            mask
                        )
                Xt_to_export[step] = pd.DataFrame(
                    np.concatenate([indeces, Xt.copy()], axis=1),
                    columns=["index"] + [l for l in labels if l != "None"],
                )

        result = pipeline[-1].fit_predict(Xt, yt)
        yt_to_export["pred"] = pd.DataFrame(
            np.concatenate([indeces, result.reshape(-1, 1).astype(int)], axis=1),
            columns=["index", "target"],
        )
        external_metric = 1 - metrics.adjusted_mutual_info_score(
            yt.reshape(1, -1).flatten(), result
        )
        metric = config["metric"]
        if "sil-" in metric:
            if Xt.shape[1] > 2:
                func = TSNE if metric.split("-")[1] == "tsne" else PCA
                Xt = func(n_components=2, random_state=42).fit_transform(Xt)
            metric = "sil"
        if "dbi-" in metric:
            if Xt.shape[1] > 2:
                func = TSNE if metric.split("-")[1] == "tsne" else PCA
                Xt = func(n_components=2, random_state=42).fit_transform(Xt)
            metric = "dbi"
        if metric == "sil":
            internal_metric = 1 - silhouette_score(Xt, result)
            # sil_samples, intra_clust_dists, inter_clust_dists = my_silhouette_samples(Xt, result)
        elif metric == "ch":
            internal_metric = -1 * calinski_harabasz_score(Xt, result)
        elif metric == "dbi":
            internal_metric = davies_bouldin_score(Xt, result)
        elif metric == "sdbw":
            internal_metric = S_Dbw(Xt, result)
        elif metric == "ssw":
            internal_metric = pipeline[-1].inertia_
        elif metric == "sw":
            _, intra_clust_dists, _ = my_silhouette_samples(Xt, result)
            internal_metric = intra_clust_dists.sum()
        elif metric == "ami":
            internal_metric = external_metric
        else:
            internal_metric = weighted_metric(
                X.copy(), Xt.copy(), result.copy(), metric, indeces
            )
        internal_metric = np.float64(internal_metric)
        external_metric = np.float64(external_metric)
        status = STATUS_OK
    except Exception as e:
        internal_metric = float("inf")
        external_metric = float("inf")
        status = STATUS_FAIL
        print(
            f"""MyException: {e}
              {traceback.print_exc()}"""
        )
    stop = time.time()

    details_path = os.path.join(config["result_path"], "details", config["file_name"])

    if not os.path.exists(details_path):
        os.makedirs(details_path)

    iteration_number = len(context["history"])
    file_name = config["file_name"] + "_" + str(iteration_number)
    try:
        for step, xt_df in Xt_to_export.items():
            xt_df.to_csv(
                os.path.join(details_path, file_name + "_X_" + step + ".csv"),
                index=False,
            )
        for step, yt_df in yt_to_export.items():
            yt_df.to_csv(
                os.path.join(details_path, file_name + "_y_" + step + ".csv"),
                index=False,
            )
    except Exception as e:
        f = open(os.path.join(details_path, file_name + ".txt"), "a+")
        f.write(str(e))
        f.close()

    item.update(
        {
            "start_time": start,
            "stop_time": stop,
            "duration": stop - start,
            "loss": internal_metric,
            "status": status,
            "internal_metric": internal_metric,
            "external_metric": external_metric,
            "iteration": iteration_number,
            "config_hash": item_hash,
            "max_history_internal_metric": context["max_history_internal_metric"],
            "max_history_external_metric": context["max_history_external_metric"],
        }
    )

    if context["max_history_internal_metric"] > internal_metric:
        item["max_history_internal_metric"] = internal_metric
        item["max_history_external_metric"] = external_metric
        context["max_history_internal_metric"] = internal_metric
        context["max_history_external_metric"] = external_metric
        context["best_config"] = item

    # Update hash index
    context["history_hash"].append(item_hash["config"])
    context["history_index"][item_hash["config"]] = iteration_number
    context["iteration"] = iteration_number

    context["history"].append(item)

    print(
        "CURRENT {}.\tinternal: {}\texternal: {}\t| BEST {}.\tinternal: {}\texternal: {}".format(
            iteration_number,
            round(item["internal_metric"], 3),
            round(item["external_metric"], 3),
            context["best_config"]["iteration"],
            round(item["max_history_internal_metric"], 3),
            round(item["max_history_external_metric"], 3),
        )
    )
    with open(os.path.join(details_path, "context.json"), "w") as outfile:
        json.dump(context, outfile, indent=4)

    return item


def objective_union(wconfig, X, y, context, config):
    return objective(wconfig["pipeline"], wconfig["algorithm"], X, y, context, config)
