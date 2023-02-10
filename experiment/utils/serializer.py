from datetime import datetime
import json
import os

def serialize_results(scenario, policy, result_path, pipeline):
    results = {
        'scenario': scenario,
        'context': policy.context,
        'pipeline': pipeline
    }
    with open(result_path, 'w') as outfile:
        json.dump(results, outfile, indent=4)