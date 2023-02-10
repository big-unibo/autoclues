from __future__ import print_function
import datetime

import itertools
import os 
import sys
import time

import pandas as pd
import yaml
from datetime import timedelta

from six import iteritems
from results_processors_utils import load_result
script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..' )
sys.path.append( mymodule_dir )
from utils import get_scenario_info, create_directory, SCENARIO_PATH, OPTIMIZATION_RESULT_PATH

def main():
    
    scenarios = get_scenario_info()

    scenario_with_results = {k: v for k, v in iteritems(scenarios) if v['results'] is not None}

    results = {}
    for _, scenario in scenario_with_results.items():
        with open(os.path.join(SCENARIO_PATH, scenario['path']), 'r') as stream:
            try:
                c = yaml.safe_load(stream)
                dataset = c['general']['dataset']
                for optimization_method, optimization_conf in c['optimizations'].items():
                    optimization_internal_metric = optimization_conf['metric']
                    #results[optimization_method] = {dataset: [optimization_internal_metric]}
                    if optimization_method not in results:
                        results[optimization_method] = {}
                    if dataset not in results[optimization_method]:
                        results[optimization_method][dataset] = []
                    if optimization_internal_metric not in results[optimization_method][dataset]:
                        results[optimization_method][dataset].append(optimization_internal_metric)
            except yaml.YAMLError as exc:
                print(exc)

    timing_results = pd.DataFrame()
    for optimization_method in results.keys():
        input_path = os.path.join(OPTIMIZATION_RESULT_PATH, optimization_method)
        for dataset, optimization_internal_metrics in results[optimization_method].items():
            for optimization_internal_metric in optimization_internal_metrics:
                df = load_result(input_path, dataset, optimization_internal_metric)
                df = df[['dataset', 'iteration', 'duration']]
                if optimization_method == 'smbo':
                    if dataset == 'synthetic':
                        df = df[df['iteration'] < 220/4]
                    if dataset == 'iris':
                        df = df[df['iteration'] < 176/4]
                    if dataset == 'breast':
                        df = df[df['iteration'] < 440/4]
                    if dataset == 'parkinsons':
                        df = df[df['iteration'] < 1012/4]
                    if dataset == 'seeds':
                        df = df[df['iteration'] < 352/4]
                    if dataset == 'wine':
                        df = df[df['iteration'] < 616/4]
                    if dataset == 'blood':
                        df = df[df['iteration'] < 176/4]
                    if dataset == 'vehicle':
                        df = df[df['iteration'] < 836/4]
                    if dataset == 'diabetes':
                        df = df[df['iteration'] < 396/4]
                    if dataset == 'appendicitis':
                        df = df[df['iteration'] < 352/4]
                    if dataset == 'ecoli':
                        df = df[df['iteration'] < 352/4]
                
                timing_results = timing_results.append(pd.DataFrame({
                    'optimization_method': [optimization_method],
                    'dataset': [dataset],
                    'metric': [optimization_internal_metric],
                    'duration': [str(timedelta(seconds=df.sum()['duration']+30))]
                }), ignore_index=True)
    timing_results.to_csv('timing.csv')
main()