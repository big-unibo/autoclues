from __future__ import print_function
import datetime

import itertools
import os
import sys
import time

import pandas as pd
import yaml

from six import iteritems
from results_processors_utils import load_result

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, "..")
sys.path.append(mymodule_dir)
from utils.utils import (
    get_scenario_info,
    create_directory,
    SCENARIO_PATH,
    OPTIMIZATION_RESULT_PATH,
)


def main():

    scenarios = get_scenario_info()

    scenario_with_results = {
        k: v for k, v in iteritems(scenarios) if v["results"] is not None
    }

    results = {}
    for file_conf, scenario in scenario_with_results.items():
        file_name = file_conf.split("__")[0]
        with open(os.path.join(SCENARIO_PATH, scenario["path"]), "r") as stream:
            try:
                c = yaml.safe_load(stream)
                dataset = c["general"]["dataset"]
                space = c["general"]["space"]
                for optimization_method, optimization_conf in c[
                    "optimizations"
                ].items():
                    optimization_internal_metric = optimization_conf["metric"]
                    budget = optimization_conf["budget"]
                    # results[optimization_method] = {dataset: [optimization_internal_metric]}
                    if optimization_method not in results:
                        results[optimization_method] = {}
                    if dataset not in results[optimization_method]:
                        results[optimization_method][dataset] = []
                    if (
                        file_name,
                        optimization_internal_metric,
                        budget,
                        space,
                    ) not in results[optimization_method][dataset]:
                        results[optimization_method][dataset].append(
                            {
                                "file_name": file_name,
                                "optimization_internal_metric": optimization_internal_metric,
                                "budget": budget,
                                "space": space,
                            }
                        )
            except yaml.YAMLError as exc:
                print(exc)

    for optimization_method in results.keys():
        input_path = os.path.join(OPTIMIZATION_RESULT_PATH, optimization_method)
        output_path = create_directory(input_path, "summary")
        output_file_name = "summary.csv"
        print(f"Summarizing {optimization_method} runs")
        try:
            summary = pd.read_csv(os.path.join(output_path, output_file_name))
            print("\tA previous summarization was found")
        except:
            print("\tA previous summarization was not found")
            print("\tSummarization process starts")
            summary = pd.DataFrame()
            start_time = time.time()
            for dataset, infos in results[optimization_method].items():
                print(f"\t\tdataset: {dataset}\n\t\t\tinternal_metrics: {infos}")
                for info in infos:
                    try:
                        summary = summary.append(
                            load_result(
                                input_path,
                                dataset=dataset,
                                info=info,
                            )
                        )
                    except Exception as e:
                        print(e)
            end_time = time.time()
            duration = int(end_time) - int(start_time)
            summary["total_duration"] = summary.groupby("dataset")[
                "duration"
            ].transform("sum")
            summary["cumsum"] = summary.groupby("dataset")["duration"].transform(
                pd.Series.cumsum
            )
            # summary['cumsum'] *= summary["budget"] / summary["total_duration"]
            summary["percentage"] = (
                summary.groupby("dataset")["iteration"].transform("max")
                / summary["tot_conf"]
            )
            # summary["cumulated_duration"] = summary["duration"].cumsum()
            print(
                f"Summarization process ends: {datetime.timedelta(seconds=duration)}\n"
            )
            # pd.read_csv("results/optimization/smbo/summary/summary.csv").groupby(["dataset"]).sum()["duration"]["iris"]
            # summary["cumulated_duration"] = summary.groupby(["dataset"])["duration"].transform("sum")
            summary.to_csv(os.path.join(output_path, output_file_name), index=False)


main()
