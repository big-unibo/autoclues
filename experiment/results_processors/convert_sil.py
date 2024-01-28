import json
import openml
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import silhouette_score


def main():
    input_path = os.path.join("results", "optimization", "smbo")
    files = [f for f in os.listdir(input_path) if '.json' in f]
    print(files)

    try:
        results = pd.DataFrame()
        for file in files:
            print(file)
            with open(os.path.join(input_path, file)) as f:
                result = json.load(f)["context"]
                name = file.split(".")[0]
                y_pred = pd.read_csv(
                        os.path.join(
                            input_path,
                            "details",
                            name,
                            "_".join(
                                [
                                    name,
                                    str(result["best_config"]["iteration"]),
                                    "y",
                                    "pred",
                                ]
                            )
                            + ".csv",
                        )
                    ).drop("index", axis="columns")
                last_trans = (
                    "outlier"
                    if result["best_config"]["pipeline"]["outlier"][0] != "outlier_NoneType"
                    else (
                        "normalize"
                        if result["best_config"]["pipeline"]["normalize"][0] != "normalize_NoneType"
                        else (
                            "features"
                            if result["best_config"]["pipeline"]["features"][0] != "features_NoneType"
                            else "original"
                        )
                    )
                )
                Xt = pd.read_csv(
                        os.path.join(
                            input_path,
                            "details",
                            name,
                            "_".join(
                                [
                                    name,
                                    str(result["best_config"]["iteration"]),
                                    "X",
                                    last_trans,
                                ]
                            )
                            + ".csv",
                        )
                    ).drop("index", axis="columns")
                # print(Xt, y_pred)
                results = pd.concat([
                    results,
                    pd.DataFrame({
                        "dataset": [name],
                        "SIL": [silhouette_score(Xt, y_pred)]
                        })
                ])
    except:
        print("No output file")
    results.to_csv(os.path.join(input_path, "comparison.csv"))

if __name__ == "__main__":
    main()