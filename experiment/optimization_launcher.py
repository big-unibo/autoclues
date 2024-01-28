import os
import subprocess
import time

from prettytable import PrettyTable
from tqdm import tqdm

from utils.utils import get_scenario_info, SCENARIO_PATH, OPTIMIZATION_RESULT_PATH
from utils import scenarios as scenarios_util
from utils import cli

scenarios = get_scenario_info()
args = cli.parse_args()

scenario_with_results = {k: v for k, v in scenarios.items() if v["results"] is not None}
t_with_results = PrettyTable(["NAME", "RESULTS"])
t_with_results.align["NAME"] = "l"
t_with_results.align["RESULTS"] = "l"
for k, v in scenario_with_results.items():
    t_with_results.add_row([k, v["results"]])

to_run = {k: v for k, v in scenarios.items() if v["results"] is None}
t_to_run = PrettyTable(["PATH"])
t_to_run.align["NAME"] = "l"
for name in to_run.keys():
    t_to_run.add_row([name])

print
print("# SCENARIOS WITH AVAILABLE RESULTS")
print(t_with_results)

print
print("# SCENARIOS TO BE RUN")
print(t_to_run)
print

with tqdm(total=len(to_run)) as pbar:
    for name, info in to_run.items():
        base_scenario = info["path"].split(".yaml")[0]
        pbar.set_description("Running scenario {}\n\r".format(name))
        print()

        current_scenario = scenarios_util.load(
            os.path.join(SCENARIO_PATH, info["path"])
        )
        file_name, optimization_approach = name.split("__")
        config = scenarios_util.to_config(
            current_scenario, file_name, optimization_approach
        )

        result_path = os.path.join(OPTIMIZATION_RESULT_PATH, optimization_approach)

        cmd = "python experiment/optimizer.py -s {} -p {} -r {} -o {} -f {}".format(
            os.path.join(SCENARIO_PATH, info["path"]),
            "features normalize outlier",
            result_path,
            optimization_approach,
            file_name,
        )
        with open(
            os.path.join(result_path, "{}_stdout.txt".format(base_scenario)), "a"
        ) as log_out:
            with open(
                os.path.join(result_path, "{}_stderr.txt".format(base_scenario)), "a"
            ) as log_err:
                start_time = time.time()
                process = subprocess.call(
                    cmd, shell=True, stdout=log_out, stderr=log_err
                )
                print("--- %s seconds ---" % (time.time() - start_time))

        pbar.update(1)
