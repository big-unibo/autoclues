import json
import openml
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

from results_processors_utils import get_dataset

def process_results(df):
    df = df[["cadence", "score"]]
    df["score"] /= df["score"].max()
    df = df.set_index("cadence")
    return df

def parse_dataset(dataset):
    if dataset == "synthetic":
        return "Synth"
    elif dataset == "blood":
        return "Blood-transfusion"
    elif dataset == "breast":
        return "Breast-tissue"
    elif dataset == "thyroid":
        return "Thyroid-new"
    else:
        return dataset.capitalize()

def exp2_plot():

    SMALL_SIZE = 15
    MEDIUM_SIZE = 18

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    df = pd.read_csv(os.path.join("results", "diversification", "timing.csv"))
    df = df[["dataset", "cadence", "score"]]
    df = df.sort_values(by=["dataset", "cadence"])

    for dataset in df["dataset"].unique():
        current_df = process_results(df[df["dataset"] == dataset])
        ax = current_df.plot(legend=False)
        fig = ax.get_figure()
        fig.savefig(os.path.join("plots", f"{dataset}.png"))

    num_rows, num_cols = 3, 4
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(num_rows, 16)
    # fig, axs = plt.subplots(num_rows, num_cols)
    axs = [
        plt.subplot(gs[0,0:4]),
        plt.subplot(gs[0,4:8]),
        plt.subplot(gs[0,8:12]),
        plt.subplot(gs[0,12:16]),
        plt.subplot(gs[1,0:4]),
        plt.subplot(gs[1,4:8]),
        plt.subplot(gs[1,8:12]),
        plt.subplot(gs[1,12:16]),
        plt.subplot(gs[2,1:5]),
        plt.subplot(gs[2,6:10]),
        plt.subplot(gs[2,11:15]),
    ]

    for idx, dataset in enumerate(["synthetic"] + [d for d in df["dataset"].unique() if d != "synthetic"]):
        current_df = process_results(df[df["dataset"] == dataset])
        # ax = axs[int(idx / num_cols), idx % num_cols]
        ax = axs[idx]
        ax.plot(current_df.index, current_df["score"].cummax())
        ax.set_xscale('log')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Norm. score")
        ax.set_title(parse_dataset(dataset))
        # ax.set_xlim([60, 7200])
        # ax.set_xticks(list(range(60, 7201, 1800)))
        ax.set_xticks([60, 300, 900, 2700, 7200])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim([0.4, 1.01])
        ax.set_yticks(list(np.arange(0.4, 1.01, 0.1)))

    # _handles, _labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(_labels, _handles))
    # lgd = fig.legend(
    #     by_label.values(),
    #     by_label.keys(),
    #     loc="upper center",
    #     ncol=3,
    #     bbox_to_anchor=(0.5, -0.13),
    # )
    # text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    # fig.set_size_inches(13, 6)
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join("plots", f"exp2.{ext}"),
            # bbox_extra_artists=(lgd, text),
            # bbox_inches="tight"
        )

def exps_table():

    df = pd.read_csv(os.path.join("results", "optimization", "smbo", "summary", "summary.csv"))
    df = df[["dataset", "iteration", "tot_conf", "percentage"]]
    df = df.sort_values(by=["dataset"])
    df = df.groupby(["dataset"]).max()
    df["percentage"] *= 100
    df = df.round(2)
    df.to_csv(os.path.join("plots", "exp2.csv"))

    df2 = pd.DataFrame()
    for dataset in df.index.unique():
        _, _, original_features = get_dataset(dataset)
        num_features = len(original_features)
        tot_confs = 44 * num_features
        explored_confs = tot_confs/4
        df2 = df2.append({
            "dataset": dataset,
            "explored_confs": int(tot_confs/4),
            "tot_confs": int(tot_confs),
            "percentage": 25
        }, ignore_index=True)
    df2.to_csv(os.path.join("plots", "exp1.csv"), index=False)

def main():

    exp2_plot()
    exps_table()

if __name__ == "__main__":
    main()