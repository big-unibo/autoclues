import os

import numpy as np
import pandas as pd

import traceback


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.normpath(path)


def get_max_performance(input_path, output_path):

    

    approaches = ["clustering"]
    get_metric = lambda approach: "ami" if approach == "clustering" else "jaccard"
    params = [60, 120, 300, 600, 900, 1800, 2700, 3600, 5400, 7200]
    datasets = [
            # "diabetes",
            # "blood",
            # "ecoli",
            # "parkinsons",
            # "seeds",
            # "breast",
            # "iris",
            # "pendigits",
            # "wine",
            # "thyroid",
            # "vehicle",
        ] + [ f"syn{idx}" for idx in range(20)]
    opt_metrics = [
        "optimization_internal_metric_value",
        "optimization_external_metric_value",
    ]

    results = pd.DataFrame()
    for approach in approaches:
        for param in params:
            for dataset in datasets:
                try:
                    results = pd.concat([
                        results,
                        pd.concat([
                            pd.DataFrame({
                                "approach": [approach]*3,
                                "time": [param]*3,
                            }),
                            pd.read_csv(
                                os.path.join(
                                    input_path,
                                    dataset,
                                    "3",
                                    f"{approach}_0-5_{get_metric(approach)}_{param}.csv",
                                )
                            )
                        ], axis=1)
                    ])
                except Exception as e:
                    print(e)

    
    results["optimization_internal_metric_value"] *= -1
    for metric in opt_metrics:
        results[metric] = round(
            1 - results[metric], 2
        )
    results = results.reset_index()
    results.to_csv(os.path.join(output_path, "raw.csv"))
    grouped = results.groupby(by=["dataset", "approach", "time"]).max()
    grouped.to_csv(os.path.join(output_path, "max_raw.csv"))
    grouped[opt_metrics].to_csv(os.path.join(output_path, "max.csv"))

    for time in [params[:(idx+1)] for idx, _ in enumerate(params)]:
        new_grouped = results[results["time"].isin(time)].groupby(by=["dataset", "approach"]).max().reset_index()
        new_grouped.to_csv(os.path.join(output_path, f"max_{time[-1]}_raw.csv"))
        new_grouped[["dataset"] + opt_metrics].to_csv(os.path.join(output_path, f"max_{time[-1]}.csv"))

    grouped_filtered = results.groupby(by=["dataset"]).max()
    grouped_filtered.to_csv(os.path.join(output_path, "max_final_raw.csv"))
    grouped_filtered[opt_metrics].to_csv(os.path.join(output_path, "max_final.csv"))



def merge_scores(input_path, output_path):

    measures = {
        "score": "Norm. score",
        "optimization_internal_metric_value": "SIL",
        "optimization_external_metric_value": "AMI",
        "timing": "Div. time"
        }

    timing = pd.read_csv(os.path.join("results", "diversification", "timing_clustering.csv"))
    performance = pd.read_csv(os.path.join("results","diversification","summary","max_raw.csv")).rename(columns={"time": "cadence"})
    scores = pd.read_csv(os.path.join("results","new_scores.csv"))
    df = scores.merge(performance, on=["dataset", "cadence"]).merge(timing[["dataset", "cadence", "timing"]], on=["dataset", "cadence"])
    df = df[["dataset", "cadence"] + list(measures.keys())]
    df = df.rename(measures, axis="columns")
    df = df.sort_values(by=["dataset", "cadence"])
    df = df.groupby("dataset").max()
    df[measures["timing"]] = df[measures["timing"]].apply(lambda val: '%.2E' % val)
    df = df.drop("cadence", axis="columns")
    df.to_csv(os.path.join(output_path, "table2.csv"))

    # for measure in measures.values():
    #     for idx in range(20):
    #         df.loc[df["dataset"] == f"syn{idx}", measure] = df.loc[df["dataset"] == f"syn{idx}", measure].cummax()



def main():

    input_path = os.path.join("/", "home", "results", "diversification", "smbo", "mmr")
    output_path = make_dir(
        os.path.join("/", "home", "results", "diversification", "summary")
    )
    # get_max_performance(input_path, output_path)
    merge_scores(input_path, output_path)



if __name__ == "__main__":
    main()
