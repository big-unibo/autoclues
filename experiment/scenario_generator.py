import yaml
import os
import copy
from tqdm import tqdm


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.normpath(path)


template = {
    "general": {"seed": 42, "space": "extended"},
    "optimizations": {
        "smbo": {"budget_kind": "time", "budget": 7200, "metric": "dbi-tsne"}
    },
    "diversifications": {
        "mmr": {
            "num_results": 3,
            "method": "mmr",
            "lambda": 0.5,
            "criterion": "clustering",
            "metric": "ami",
        },
        "exhaustive": {
            "num_results": 3,
            "method": "mmr",
            "lambda": 0.5,
            "criterion": "clustering",
            "metric": "ami",
        },
    },
    "runs": ["smbo_mmr"],
}

if __name__ == "__main__":
    input_path = make_dir(os.path.join("/", "home", "scenarios"))

    datasets = [
            "avila",
            "diabetes",
            "isolet",
            "postures",
            "blood",
            "ecoli",
            "parkinsons",
            "seeds",
            "breast",
            "iris",
            "pendigits",
            "statlog",
            "synthetic",
            "wine",
            "thyroid",
            "vehicle",
        ] + [ f"syn{idx}" for idx in range(20)]

    with tqdm(total=len(datasets)) as pbar:
        for dataset in datasets:
            task_template = copy.deepcopy(template)
            task_template["general"]["dataset"] = dataset
            with open(
                os.path.join(
                    input_path,
                    f"""{dataset}.yaml""",
                ),
                "w",
            ) as f:
                yaml.dump(task_template, f)
            pbar.update()
