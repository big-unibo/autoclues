import yaml


def load(path):
    scenario = None
    with open(path, "r") as f:
        try:
            scenario = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            scenario = None
    if scenario is not None:
        scenario["file_name"] = path.split("/")[-1].split(".")[0]
    return scenario


def validate(scenario):
    return True  #  TODO


def to_config(scenario, file_name, optimization_method):
    config = {
        "file_name": file_name,
        "dataset": scenario["general"]["dataset"],
        "seed": scenario["general"]["seed"],
        "space": scenario["general"]["space"],
        "metric": scenario["optimizations"][optimization_method]["metric"],
        "budget_kind": scenario["optimizations"][optimization_method]["budget_kind"],
        "budget": scenario["optimizations"][optimization_method]["budget"],
    }
    return config
