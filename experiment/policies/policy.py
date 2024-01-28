import math
from pipeline.PrototypeSingleton import PrototypeSingleton

import json


class Policy(object):
    def __init__(self, config):
        self.PIPELINE_SPACE = PrototypeSingleton.getInstance().getDomainSpace()
        self.max_k = int(math.sqrt(PrototypeSingleton.getInstance().getX().shape[0]))
        self.compute_baseline = False
        self.config = config
        self.context = {
            "iteration": 0,
            "history_hash": [],
            "history_index": {},
            "history": [],
            "max_history_internal_metric": float("inf"),
            "max_history_external_metric": float("inf"),
            "best_config": {},
        }

    def __compute_baseline(self, X, y):
        raise Exception("No implementation for baseline score")

    def run(self, X, y):
        if self.compute_baseline:
            self.__compute_baseline(X, y)

    def display_step_results(self, best_config):
        print("BEST PIPELINE:")
        print("#" * 50)
        print(f"""{json.dumps(best_config['pipeline'], indent=4, sort_keys=True)}""")
        print("#" * 50 + "\n")
        print("BEST ALGORITHM:")
        print("#" * 50)
        print(f"""{json.dumps(best_config['algorithm'], indent=4, sort_keys=True)}""")
        print("#" * 50 + "\n")
        print("BEST METRICS:")
        print("#" * 50)
        print(
            "\tINTERNAL:\t{}\n\tEXTERNAL:\t{}".format(
                round(best_config["internal_metric"], 3),
                round(best_config["external_metric"], 3),
            )
        )
        print("#" * 50)
