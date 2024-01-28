import json
import openml
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

def normalize_df(df, way):
    if way == "max":
        df /= df.max()
    elif way == "min_max":
        df-= df.min()
        df /= (df.max() - df.min())
    return df

def process_results(df, measure, normalize):
    df = df[["cadence", measure]]
    df[measure] = normalize_df(df[measure], normalize)
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

def new_exp2_plot():

    SMALL_SIZE = 15
    MEDIUM_SIZE = 18

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    timing = pd.read_csv(os.path.join("results", "diversification", "timing_clustering.csv"))
    performance = pd.read_csv(os.path.join("results","diversification","summary","max_raw.csv"))
    scores = pd.read_csv(os.path.join("results","new_scores.csv"))
    df = scores[["dataset", "cadence","score"]]
    df = df.sort_values(by=["dataset", "cadence"])


    for dataset in df["dataset"].unique():
        current_df = process_results(df[df["dataset"] == dataset], "score", None)
        ax = current_df.plot(legend=False)
        fig = ax.get_figure()
        fig.savefig(os.path.join("plots", f"{dataset}.png"))

    num_rows, num_cols = 4, 5
    fig, axs = plt.subplots(num_rows, num_cols)

    # num_rows, num_cols = 3, 5
    # fig = plt.figure(figsize=(18, 9))
    # gs = gridspec.GridSpec(num_rows, 16)
    # # fig, axs = plt.subplots(num_rows, num_cols)
    # axs = [
    #     plt.subplot(gs[0,0:4]),
    #     plt.subplot(gs[0,4:8]),
    #     plt.subplot(gs[0,8:12]),
    #     plt.subplot(gs[0,12:16]),
    #     plt.subplot(gs[1,0:4]),
    #     plt.subplot(gs[1,4:8]),
    #     plt.subplot(gs[1,8:12]),
    #     plt.subplot(gs[1,12:16]),
    #     plt.subplot(gs[2,1:5]),
    #     plt.subplot(gs[2,6:10]),
    #     plt.subplot(gs[2,11:15]),
    # ]

    print(df[df["dataset"] == "syn13"])

    # for idx, dataset in enumerate([d for d in df["dataset"].unique() if d.startswith("syn") and d != "synthetic"]):
        # current_df = process_results(df[df["dataset"] == dataset])
        # ax = axs[int(idx / num_cols), idx % num_cols]

        # ax = axs[idx]
    for idx in range(20):
        current_df = process_results(df[df["dataset"] == f"syn{idx}"], "score", None)

        ax = axs[int(idx / num_cols), idx % num_cols]
        ax.plot(current_df.index, current_df["score"].cummax())
        ax.set_xscale('log')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Norm. score")
        # ax.set_title(parse_dataset(dataset))
        ax.set_title(f"syn{idx+1}")
        # ax.set_xlim([60, 7200])
        # ax.set_xticks(list(range(60, 7201, 1800)))
        ax.set_xticks([60, 300, 900, 2700, 7200])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim([0, 1.01])
        # ax.set_yticks(list(np.arange(0.4, 1.01, 0.1)))

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
    # fig.tight_layout()
    # fig.set_size_inches(13, 6)
    fig.set_size_inches(30, 30)
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join("plots", f"exp2.{ext}"),
            # bbox_extra_artists=(lgd, text),
            # bbox_inches="tight"
        )


def new_plot_raw(normalize="none"):

    SMALL_SIZE = 15
    MEDIUM_SIZE = 18

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    measures = ["score", "optimization_internal_metric_value", "optimization_external_metric_value"]

    timing = pd.read_csv(os.path.join("results", "diversification", "timing_clustering.csv"))
    performance = pd.read_csv(os.path.join("results","diversification","summary","max_raw.csv")).rename(columns={"time": "cadence"})
    scores = pd.read_csv(os.path.join("results","new_scores.csv"))
    df = scores.merge(performance, on=["dataset", "cadence"])
    df = df[["dataset", "cadence"] + measures]
    df = df.sort_values(by=["dataset", "cadence"])

    df.to_csv("trial.csv")
    for measure in measures:
        num_rows, num_cols = 4, 5
        fig, axs = plt.subplots(num_rows, num_cols)
        for idx in range(20):
            current_df = process_results(df[df["dataset"] == f"syn{idx}"], measure, normalize)
            ax = axs[int(idx / num_cols), idx % num_cols]
            ax.plot(current_df.index, current_df[measure].cummax())
            ax.set_xscale('log')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Norm. score")
            ax.set_title(f"syn{idx+1}")
            ax.set_xticks([60, 300, 900, 2700, 7200])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_ylim([0, 1.01])
        fig.set_size_inches(30, 30)
        for ext in ["png", "pdf"]:
            fig.savefig(
                os.path.join("plots", f"{measure}_{normalize}.{ext}"),
            )

def set_figure(fig, num_plots, name):
    fig.set_size_inches(5*num_plots, 4)
    for ext in ["png", "pdf"]:
        fig.savefig(f"{name}.{ext}", bbox_inches='tight')


def new_plot_agg(normalize="none"):

    SMALL_SIZE = 16
    MEDIUM_SIZE = 20

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    rel_measures ={
        "optimization_internal_metric_value": "SIL",
        "optimization_external_metric_value": "AMI",
    }
    div_measures = {
        "score": "Score",
        "timing": "Div. time"
        }
    measures = {**rel_measures, **div_measures}

    timing = pd.read_csv(os.path.join("results", "diversification", "timing_clustering.csv"))
    performance = pd.read_csv(os.path.join("results","diversification","summary","max_raw.csv")).rename(columns={"time": "cadence"})
    scores = pd.read_csv(os.path.join("results","new_scores.csv"))
    df = scores.merge(performance, on=["dataset", "cadence"]).merge(timing[["dataset", "cadence", "timing"]], on=["dataset", "cadence"])

    df = df[["dataset", "cadence"] + list(measures.keys())]
    df = df.rename(measures, axis="columns")
    df = df.sort_values(by=["dataset", "cadence"])

    df.to_csv("trial.csv")


    for measure in measures.values():
        for idx in range(20):
            df.loc[df["dataset"] == f"syn{idx}", measure] = normalize_df(
                df.loc[df["dataset"] == f"syn{idx}", measure].cummax(),
                normalize
            )
    df = df.groupby("cadence").agg({column: {"mean": np.mean, "std": np.std} for column in measures.values()})
    df.columns = df.columns.to_flat_index().str.join('_')


    for file_name, new_measures in {"rel": rel_measures, "div": div_measures, "all": measures}.items():
        num_rows, num_cols = 1, len(new_measures)
        fig, axs = plt.subplots(num_rows, num_cols)
        for idx, measure in enumerate(new_measures.values()):
            ax = axs[idx]
            color_idx = idx  + (0 if file_name != "div" else 2)
            ax.plot(df.index, df[f"{measure}_mean"], color=f"C{color_idx}", label="mean")
            ax.fill_between(
                df.index,
                df[f"{measure}_mean"] - df[f"{measure}_std"],
                df[f"{measure}_mean"] + df[f"{measure}_std"],
                alpha=0.5,
                color=f"C{color_idx}",
                label="std"
            )
            ax.set_xscale('log')
            ax.set_xlabel("Time (s)")
            ax.set_title(measure)
            ax.set_xticks([60, 300, 900, 2700, 7200])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_ylim([0, 1.05])
            ax.set_xlim([60, 7200])
            ax.legend(
                ncol=1,
                # title=measure,
                loc='lower right')
        set_figure(
            fig=fig,
            num_plots=len(new_measures),
            name=os.path.join("plots", file_name)
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

    # exp2_plot()
    # exps_table()
    new_plot_agg(normalize="max")

if __name__ == "__main__":
    main()