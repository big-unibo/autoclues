import json
import openml
import os

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from pipeline.PrototypeSingleton import PrototypeSingleton
from utils import scenarios, serializer, cli, datasets
import policies
from sklearn.model_selection import train_test_split


def main(args):
    scenario = scenarios.load(args.scenario)
    config = scenarios.to_config(scenario, args.file_name, args.optimization_approach)
    with open(os.path.join("resources", f"""{config["space"]}_space.json"""), "r") as f:
        space = json.load(f)

    print("SCENARIO:")
    print("#" * 50)
    print(f"{json.dumps(scenario, indent=4, sort_keys=True)}")
    print("#" * 50 + "\n")

    np.random.seed(config["seed"])

    X, y, _ = datasets.get_dataset(config["dataset"])
    PrototypeSingleton.getInstance().setPipeline(args.pipeline, space)

    config["result_path"] = args.result_path
    policy = policies.initiate(config)
    print(config)
    policy.run(X, y, space)

    serializer.serialize_results(
        scenario,
        policy,
        os.path.join(args.result_path, config["file_name"] + ".json"),
        args.pipeline,
    )


if __name__ == "__main__":
    args = cli.parse_args()
    main(args)
