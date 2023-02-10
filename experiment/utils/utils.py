import os
import yaml

from six import iteritems

from utils import scenarios as scenarios_util

RESULT_PATH = 'results'
SCENARIO_PATH = 'scenarios'

def create_directory(result_path, directory):
    result_path = os.path.join(result_path, directory)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path

OPTIMIZATION_RESULT_PATH = create_directory(RESULT_PATH, 'optimization')
DIVERSIFICATION_RESULT_PATH = create_directory(RESULT_PATH, 'diversification')

def get_scenario_info():
    # Gather list of scenarios
    scenario_list = [p for p in os.listdir(SCENARIO_PATH) if '.yaml' in p]
    scenario_map = {}

    # Determine which one have no result files
    for scenario in scenario_list:
        base_scenario = scenario.split('.yaml')[0]
        scenario_configurations = scenarios_util.load(os.path.join(SCENARIO_PATH, scenario))
        for optimization_method, configuration in scenario_configurations['optimizations'].items():
            if optimization_method != ('exhaustive' if configuration['budget'] == 'inf' else 'smbo'):
                raise Exception('Optimization method name different to what declared in the related configuration')
            complete_scenario_name = base_scenario + '__' + optimization_method
            if complete_scenario_name not in scenario_map.keys():
                scenario_map[complete_scenario_name] = {'results': None, 'path': scenario}
                relative_result_path = create_directory(OPTIMIZATION_RESULT_PATH, optimization_method)
                result_list = [p for p in os.listdir(relative_result_path) if '.json' in p]
                for result in result_list:
                    base_result = result.split('.json')[0]
                    if base_result.__eq__(base_scenario):
                        scenario_map[complete_scenario_name]['results'] = os.path.join(relative_result_path, result)
    return scenario_map