import datetime
import os
import sys
import statistics
import yaml
import warnings
import itertools
import time
import argparse
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.manifold import TSNE
from matplotlib import cm
from scipy.spatial import distance
from six import iteritems
script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..' )
sys.path.append( mymodule_dir )
from utils.utils import create_directory, get_scenario_info, SCENARIO_PATH, OPTIMIZATION_RESULT_PATH, DIVERSIFICATION_RESULT_PATH
from pipeline.outlier_detectors import LocalOutlierDetector
from utils import datasets

def get_last_transformation(df, dataset, optimization_internal_metric, iteration):
    pipeline, is_there = {}, {}
    for transformation in ['features', 'normalize', 'outlier']:
        pipeline[transformation] = df.loc[(
            (df['dataset'] == dataset) & 
            (df['optimization_internal_metric'] == optimization_internal_metric) & 
            (df['iteration'] == iteration)), transformation].values[0]
        is_there[transformation] = pipeline[transformation] != 'None'
    last_transformation = 'outlier' if is_there['outlier'] else ('normalize' if is_there['normalize'] else ('features' if is_there['features'] else 'original'))
    return pipeline, last_transformation

def get_one_hot_encoding(current_features, original_features):
    features_to_return = []
    for feature in original_features:
        features_to_return.append(1 if feature in current_features else 0)
    return features_to_return

def diversificate_mmr(meta_features, conf, original_features):
    working_df = meta_features.copy()
    working_df = working_df.sort_values(by=['optimization_internal_metric_value'], ascending=False)
    first_solution = working_df.iloc[0]
    solutions = pd.DataFrame()
    solutions = solutions.append(first_solution)
    working_df = working_df.drop([first_solution.name])

    for _ in range(conf['diversification_num_results']-1):
        confs = working_df['iteration'].to_list()
        mmr = pd.DataFrame()
        for current_iteration in confs:
            current_optimization_internal_metric_value = float(working_df[working_df['iteration'] == current_iteration]['optimization_internal_metric_value'])
            if conf['diversification_criterion'] == 'clustering':
                current_y_pred = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['optimization_internal_metric'], str(current_iteration), 'y', 'pred']) + '.csv'))
            elif conf['diversification_criterion'] == 'features_set' or  conf['diversification_criterion'] == 'features_set_n_clusters':
                _, current_last_transformation = get_last_transformation(meta_features, conf['dataset'], conf['optimization_internal_metric'], current_iteration)
                current_X = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['optimization_internal_metric'], str(current_iteration), 'X', current_last_transformation]) + '.csv'))
                current_features = list(current_X.columns)
                current_features = get_one_hot_encoding(current_features, original_features)
                if conf['diversification_criterion'] == 'features_set_n_clusters':
                    current_features.append(meta_features.loc[(
                        (meta_features['dataset'] == conf['dataset']) & 
                        (meta_features['optimization_internal_metric'] == conf['optimization_internal_metric']) & 
                        (meta_features['iteration'] == current_iteration)), 'algorithm__n_clusters'].values[0])
            elif conf['diversification_criterion'] == 'hyper_parameter':
                _, current_last_transformation = get_last_transformation(meta_features, conf['dataset'], conf['optimization_internal_metric'], current_iteration)
                current_X = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['optimization_internal_metric'], str(current_iteration), 'X', current_last_transformation]) + '.csv'))
                current_features = [len(list(current_X.columns))]
                current_features.append(meta_features.loc[(
                    (meta_features['dataset'] == conf['dataset']) & 
                    (meta_features['optimization_internal_metric'] == conf['optimization_internal_metric']) & 
                    (meta_features['iteration'] == current_iteration)), 'algorithm__n_clusters'].values[0])
            else:
                raise Exception(f'''missing diversification criterion for 
                                {conf}''')
            others = solutions['iteration'].to_list()
            distances = []
            for other in others:
                if conf['diversification_criterion'] == 'clustering':
                    other_y_pred = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['optimization_internal_metric'], str(int(other)), 'y', 'pred']) + '.csv'))
                    if conf['diversification_metric'] == 'ami':
                        dist = 1- metrics.adjusted_mutual_info_score( current_y_pred.to_numpy().reshape(-1), other_y_pred.to_numpy().reshape(-1))
                    else:
                        raise Exception(f'''missing diversification metric for 
                                        {conf}''')
                elif conf['diversification_criterion'] == 'features_set' or  conf['diversification_criterion'] == 'features_set_n_clusters': 
                    _, other_last_transformation = get_last_transformation(meta_features, conf['dataset'], conf['optimization_internal_metric'], int(other))
                    other_X = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['optimization_internal_metric'], str(int(other)), 'X', other_last_transformation]) + '.csv'))
                    other_features = list(other_X.columns)
                    other_features = get_one_hot_encoding(other_features, original_features)
                    if conf['diversification_criterion'] == 'features_set_n_clusters':
                        other_features.append(meta_features.loc[(
                            (meta_features['dataset'] == conf['dataset']) & 
                            (meta_features['optimization_internal_metric'] == conf['optimization_internal_metric']) & 
                            (meta_features['iteration'] == int(other))), 'algorithm__n_clusters'].values[0])
                    if conf['diversification_metric'] == 'euclidean':
                        dist = distance.euclidean(current_features, other_features)
                    elif conf['diversification_metric'] == 'cosine':
                        dist = distance.cosine(current_features, other_features)
                    elif conf['diversification_metric'] == 'jaccard':
                        dist = distance.jaccard(current_features, other_features)
                    else:
                        raise Exception(f'''missing diversification metric for 
                                        {conf}''')
                elif conf['diversification_criterion'] == 'hyper_parameter':
                    _, other_last_transformation = get_last_transformation(meta_features, conf['dataset'], conf['optimization_internal_metric'], int(other))
                    other_X = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['optimization_internal_metric'], str(int(other)), 'X', other_last_transformation]) + '.csv'))
                    other_features = [len(list(other_X.columns))]
                    other_features.append(meta_features.loc[(
                        (meta_features['dataset'] == conf['dataset']) & 
                        (meta_features['optimization_internal_metric'] == conf['optimization_internal_metric']) & 
                        (meta_features['iteration'] == int(other))), 'algorithm__n_clusters'].values[0])

                    if conf['diversification_metric'] == 'euclidean':
                        dist = distance.euclidean(current_features, other_features)
                    elif conf['diversification_metric'] == 'cosine':
                        dist = distance.cosine(current_features, other_features)
                    elif conf['diversification_metric'] == 'jaccard':
                        dist = distance.jaccard(current_features, other_features)
                    else:
                        raise Exception(f'''missing diversification metric for 
                                        {conf}''')
                else:
                    raise Exception(f'''missing diversification criterion for 
                                    {conf}''')
                distances.append(dist)
            current_mean_distance = statistics.mean(distances)
            current_mmr = (1 - conf['diversification_lambda']) * current_optimization_internal_metric_value + conf['diversification_lambda'] * current_mean_distance
            mmr = mmr.append({'iteration': current_iteration, 'mmr': current_mmr}, ignore_index=True)
        mmr = mmr.sort_values(by=['mmr'], ascending=False)
        winning_conf = mmr.iloc[0]['iteration']
        winner = working_df.loc[working_df['iteration'] == winning_conf]
        solutions = solutions.append(winner)
        working_df = working_df.drop(winner.index)
    dashboard_score = evaluate_dashboard(solutions.copy(), conf, original_features)
    return {'solutions': solutions, 'score': dashboard_score}

def diversificate_exhaustive(meta_features, conf, original_features):
    working_df = meta_features.copy()
    cc = list(itertools.combinations(list(working_df.index), conf['diversification_num_results']))
    print(f'\t\t\tNum combinations to evaluate: {len(cc)}')
    print(f'\t\t\tEstimated time: {datetime.timedelta(seconds=len(cc))}')
    exhaustive_search = pd.DataFrame()
    best_dashboard = {'solutions': pd.DataFrame(), 'score': 0.}
    for c in cc:
        solutions = working_df.loc[c, :].copy()
        dashboard_score = evaluate_dashboard(solutions, conf, original_features)
        if dashboard_score > best_dashboard['score']:
            best_dashboard['solutions'] = solutions.copy()
            best_dashboard['score'] = dashboard_score
        dashboard = {}
        for i in range(conf['diversification_num_results']):
            dashboard['conf_' + str(i)] = int(solutions.loc[c[i], 'iteration'])
        dashboard['dashboard_score'] = dashboard_score
        exhaustive_search = exhaustive_search.append(dashboard, ignore_index=True)
    exhaustive_search.sort_values(by=['dashboard_score'], ascending=False)
    exhaustive_search.to_csv(os.path.join(conf['output_path'], conf['output_file_name'] + '_all' + '.csv'), index=False)
    return best_dashboard

def evaluate_dashboard(solutions, conf, original_features):
    def compute_pairwise_div(df, conf, original_features):
        df = df.reset_index()
        first_iteration = int(df.loc[0, 'iteration'])
        second_iteration = int(df.loc[1, 'iteration'])
        div_vectors = []
        for iteration in [first_iteration, second_iteration]:
            if conf['diversification_criterion'] == 'clustering':
                y_pred = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['optimization_internal_metric'], str(iteration), 'y', 'pred']) + '.csv'))
                div_vectors.append(y_pred)
            elif conf['diversification_criterion'] == 'features_set' or conf['diversification_criterion'] == 'features_set_n_clusters':
                _, last_transformation = get_last_transformation(df, conf['dataset'], conf['optimization_internal_metric'], iteration)
                X = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['optimization_internal_metric'], str(iteration), 'X', last_transformation]) + '.csv'))
                features = list(X.columns)
                features = get_one_hot_encoding(features, original_features)
                if conf['diversification_criterion'] == 'features_set_n_clusters':
                    features.append(df.loc[df['iteration'] == iteration, 'algorithm__n_clusters'].values[0])
                div_vectors.append(features)
            elif conf['diversification_criterion'] == 'hyper_parameter':
                _, last_transformation = get_last_transformation(df, conf['dataset'], conf['optimization_internal_metric'], iteration)
                X = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['optimization_internal_metric'], str(iteration), 'X', last_transformation]) + '.csv'))
                features = [len(list(X.columns))]
                features.append(df.loc[df['iteration'] == iteration, 'algorithm__n_clusters'].values[0])
                div_vectors.append(features)
            else:
                raise Exception(f'''missing diversification criterion for 
                                {conf}''')
        if conf['diversification_metric'] == 'ami':
            return 1 - metrics.adjusted_mutual_info_score(div_vectors[0].to_numpy().reshape(-1), div_vectors[1].to_numpy().reshape(-1))
        elif conf['diversification_metric'] == 'euclidean':
            return distance.euclidean(div_vectors[0], div_vectors[1])
        elif conf['diversification_metric'] == 'cosine':
            return distance.cosine(div_vectors[0], div_vectors[1])
        elif conf['diversification_metric'] == 'jaccard':
            return distance.jaccard(div_vectors[0], div_vectors[1])
        else:
            raise Exception(f'''missing diversification metric for 
                            {conf}''')
    sim = solutions['optimization_internal_metric_value'].sum()
    cc = list(itertools.combinations(list(solutions.index), 2))
    div = sum([compute_pairwise_div(solutions.loc[c, :].copy(), conf, original_features) for c in cc])
    return ((conf['diversification_num_results'] - 1) * (1 - conf['diversification_lambda']) * sim) + (2 * conf['diversification_lambda'] * div)

def save_figure(solutions, conf):
    print('\t\tPlotting process starts')
    start_time = time.time()
    fig = plt.figure(figsize=(32, 18)) 
    i = 0  
    for _, row in solutions.iterrows():
        i += 1
        pipeline, last_transformation = get_last_transformation(solutions.copy(), conf['dataset'], conf['optimization_internal_metric'], int(row['iteration']))
        
        Xt = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['optimization_internal_metric'], str(int(row['iteration'])), 'X', last_transformation]) + '.csv'))
        yt = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['optimization_internal_metric'], str(int(row['iteration'])), 'y', 'pred']) + '.csv'))
        
        if Xt.shape[1] < 3:
            ax = fig.add_subplot(3, 3, i)
        else:
            #ax = fig.add_subplot(3, 3, i, projection='3d')
            ax = fig.add_subplot(3, 3, i)
        colors = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan', 'indigo', 'black'])
        old_X = Xt.copy()
        if Xt.shape[1] > 2:
            Xt = pd.DataFrame(TSNE(n_components=2, random_state=42).fit_transform(Xt.to_numpy(), yt.to_numpy()), columns=['TSNE_0', 'TSNE_1'])
            if conf['outlier']:
                Xt, yt = LocalOutlierDetector(n_neighbors=32).fit_resample(Xt.to_numpy(), yt.iloc[:, 0].to_numpy())
                Xt, yt = pd.DataFrame(Xt, columns=['TSNE_0', 'TSNE_1']), pd.DataFrame(yt, columns=['target'])
        n_selected_features = Xt.shape[1]
        Xt = Xt.iloc[:, :n_selected_features]
        min, max = Xt.min().min(), Xt.max().max()
        range = (max-min)/10
        xs = Xt.iloc[:, 0]
        ys = [(max+min)/2] * Xt.shape[0] if n_selected_features < 2 else Xt.iloc[:, 1]
        #zs = [(max+min)/2] * Xt.shape[0] if n_selected_features < 3 else Xt.iloc[:, 2]
        if Xt.shape[1] < 3:
            ax.scatter(xs, ys, c=[colors[int(i)] for i in yt.iloc[:, 0].to_numpy()])
        #else:
            #ax.scatter(xs, ys, zs, c=[colors[int(i)] for i in yt.iloc[:, 0].to_numpy()])
        ax.set_xlim([min - range, max + range])
        ax.set_ylim([min - range, max + range])
        ax.set_xlabel(list(Xt.columns)[0], fontsize=16)
        ax.set_ylabel('None' if n_selected_features < 2 else list(Xt.columns)[1], fontsize=16)
        #if Xt.shape[1] >= 3:
            #ax.set_zlim([min, max])
            #ax.set_zlabel('None' if n_selected_features < 3 else list(Xt.columns)[2], fontsize=16)
        title = '\n'.join([operator for operator in pipeline.values() ])
        current_solution = solutions.loc[(
            (solutions['dataset'] == conf['dataset']) & 
            (solutions['optimization_internal_metric'] == conf['optimization_internal_metric']) & 
            (solutions['iteration'] == int(row['iteration']))), :]
        k_features = '\nk= ' + str(old_X.shape[1])
        n_clusters = '    n=' + str(int(current_solution.loc[:, 'algorithm__n_clusters'].values[0]))
        title += k_features + n_clusters
        title += '\nint_metr=' + str(round(current_solution.loc[:, 'optimization_internal_metric_value'].values[0], 2))
        title += '    ext_metr=' + str(round(current_solution.loc[:, 'optimization_external_metric_value'].values[0], 2))
        ax.set_title(title, fontdict={'fontsize': 20, 'fontweight': 'medium'})
    plt.tight_layout(rect=[0., 0., 1., 0.85])
    end_time = time.time()
    plotting_duration = int(end_time) - int(start_time)
    plotting_duration = str(datetime.timedelta(seconds=plotting_duration))
    print(f'\t\tPlotting process ends: {plotting_duration}')
    title = f'''{conf['dataset']}
            OPTIMIZATION method: {conf['optimization_method']}, metric: {conf['optimization_internal_metric']} 
            DIVERSIFICATION method: {conf['diversification_method']}, lambda: {conf['diversification_lambda']}, criterion: {conf['diversification_criterion']}, metric: {conf['diversification_metric']}
            DASHABOARD score: {round(conf['dashboard_score'], 2)}, div_time: {conf['diversification_duration']}, plot_time: {plotting_duration}'''
    fig.suptitle(title, fontsize=30)
    fig.savefig(os.path.join(conf['output_path'], conf['output_file_name'] + ('_outlier' if conf['outlier'] else '') + '.pdf'))

def parse_args():

    parser = argparse.ArgumentParser(description='AutoClues')

    parser.add_argument("-exp", "--experiment", nargs="?", type=str, required=False,
                        help="experiment type")

    parser.add_argument("-cad", "--cadence", nargs="?", type=int, required=False,
                        help="cadence of the second experiment")

    parser.add_argument("-max", "--max_time", nargs="?", type=int, required=False,
                        help="max time of the second experiment")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    scenarios = get_scenario_info()

    scenario_with_results = {k: v for k, v in iteritems(scenarios) if v['results'] is not None}
    paths = set([v['path'] for v in scenario_with_results.values()])
    confs = []
    for scenario in paths:
        with open(os.path.join(SCENARIO_PATH, scenario), 'r') as stream:
            try:
                c = yaml.safe_load(stream)
                for run in c['runs']:
                    optimization_method = run.split('_')[0]
                    diversification_method = run.split('_')[1]
                    conf = {
                    'dataset': c['general']['dataset'],
                    'space': c['general']['space'],
                    'optimization_method': optimization_method,
                    'optimization_internal_metric': c['optimizations'][optimization_method]['metric'],
                    'diversification_num_results': c['diversifications'][diversification_method]['num_results'],
                    'diversification_method': diversification_method,
                    'diversification_lambda': c['diversifications'][diversification_method]['lambda'],
                    'diversification_criterion': c['diversifications'][diversification_method]['criterion'],
                    'diversification_metric': c['diversifications'][diversification_method]['metric'],
                    }
                    if args.experiment == "exp2":
                        for current_time in range(args.cadence, args.max_time + args.cadence, args.cadence):
                            conf["time"] = current_time
                            confs.append(conf.copy())
                    else:
                        confs.append(conf.copy())
            except yaml.YAMLError as exc:
                print(exc)

    timing_df = pd.DataFrame()

    for i in range(len(confs)):
        conf = confs[i]
        print(f'''{i+1}th conf out of {len(confs)}: {conf}''')

        working_folder = conf['dataset'] + '_' + conf['optimization_internal_metric']
        conf['diversification_path'] = os.path.join(DIVERSIFICATION_RESULT_PATH, conf['optimization_method'], conf['diversification_method'], working_folder)
        conf['output_path'] = create_directory(conf['diversification_path'], str(conf['diversification_num_results']))
        conf['output_file_name'] = f'''{conf['diversification_criterion']}_0-{int(conf['diversification_lambda']*10)}_{conf['diversification_metric']}'''
        if args.experiment == "exp2":
            conf['output_file_name'] += f'''_{conf['time']}'''
        optimization_path = os.path.join(OPTIMIZATION_RESULT_PATH, conf['optimization_method'])
        conf['input_path'] = os.path.join(optimization_path, 'details', working_folder)

        print('\tLoading optimization process solutions')
        meta_features = pd.read_csv(os.path.join(optimization_path, 'summary', 'summary.csv'))
        meta_features = meta_features[(meta_features['dataset'] == conf['dataset']) & (meta_features['optimization_internal_metric'] == conf['optimization_internal_metric'])]

        _, _, original_features = datasets.get_dataset(conf['dataset'])
        num_features = len(original_features)

        if args.experiment == "exp1":
            # if it is smbo, we keep just the 25% of all the configurations
            if conf['optimization_method'] == 'smbo':
                mul_fact = 44 if conf['space'] == 'toy' else 2310
                tot_conf = mul_fact * (num_features if conf['space'] == 'toy' else (1 + 4*(num_features-1)))
                meta_features = meta_features[meta_features['iteration'] < tot_conf/4]
        else:
            if conf['optimization_method'] == 'smbo':
                meta_features = meta_features[meta_features['cumsum'] <= conf['time']]

        print(f'\t\tGot {meta_features.shape[0]} solutions')
        print('\t\tFiltering..')

        # meta_features1 = meta_features[meta_features['features__k'] == 'None']
        # meta_features2 = meta_features[meta_features['features__k'] != 'None']
        # meta_features2 = meta_features2[meta_features2['features__k'].astype(np.int32) < len(original_features)]
        # meta_features = pd.concat([meta_features1, meta_features2], ignore_index=True)

        # meta_features = meta_features[(meta_features['normalize'] == 'normalize_StandardScaler') | (meta_features['normalize'] == 'None')]
        # if 'normalize__with_mean' in list(meta_features.columns):
        #     meta_features = meta_features[(meta_features['normalize__with_mean'] == 'None') | (meta_features['normalize__with_mean'] == 'True')]
        # if 'normalize__with_std' in list(meta_features.columns):
        #     meta_features = meta_features[(meta_features['normalize__with_std'] == 'None') | (meta_features['normalize__with_std'] == 'True')]
        meta_features = meta_features[~((meta_features['normalize'] != 'None') & (meta_features['features__k'] == '1'))]

        # if conf['diversification_criterion'] == 'clustering':
        #     meta_features = meta_features[meta_features['outlier'] == 'None']
        # elif conf['diversification_criterion'] == 'features_set' or conf['diversification_criterion'] == 'features_set_n_clusters' or conf['diversification_criterion'] == 'hyper_parameter':
        #     meta_features = meta_features[(meta_features['outlier'] == 'None') | ((meta_features['outlier'] != 'None') & (meta_features['outlier__n_neighbors'] == ('100' if conf['dataset'] == 'synthetic' else '32')))]
        # else:
        #     raise Exception(f'''missing diversification criterion for
        #                     {conf}''')

        if conf['optimization_internal_metric'] == 'sdbw':
            meta_features['optimization_internal_metric_value'] *= -1
            meta_features['optimization_internal_metric_value'] = 1 - meta_features['optimization_internal_metric_value']
            meta_features['max_optimization_internal_metric_value'] *= -1
            meta_features['max_optimization_internal_metric_value'] = 1 - meta_features['max_optimization_internal_metric_value']
        if conf['optimization_internal_metric'] == 'sil' or conf['optimization_internal_metric'] == 'sdbw':
            metric_threshold = 0.5 if args.experiment == "exp1" else 0.01
            meta_features = meta_features[meta_features['optimization_internal_metric_value'] >= metric_threshold]
        print(f'\t\tGot {meta_features.shape[0]} solutions')
        print('\tDiversification')
        try:
            dashboard = {}
            dashboard['solutions'] = pd.read_csv(os.path.join(conf['output_path'], conf['output_file_name'] + '.csv'))
            print('\t\tA previous dashboard was found')
            print('\t\tCalculating dashboard score')
            dashboard['score'] = evaluate_dashboard(dashboard['solutions'], conf, original_features)
        except:
            print('\t\tA previous dashboard was not found')
            print('\t\tDiversification process starts')
            start_time = time.time()
            if conf['diversification_method'] == 'mmr':
                dashboard = diversificate_mmr(meta_features, conf, original_features)
            elif conf['diversification_method'] == 'exhaustive':
                dashboard = diversificate_exhaustive(meta_features, conf, original_features)
            else:
                raise Exception(f'''missing diversification method for {conf}''')
            end_time = time.time()
            conf['diversification_duration_s'] = int(end_time) - int(start_time)
            conf['diversification_duration'] = str(datetime.timedelta(seconds=conf['diversification_duration_s']))
            print(f'''\t\tDiversification process ends: {conf['diversification_duration']}''')
            dashboard['solutions'].to_csv(os.path.join(conf['output_path'], conf['output_file_name'] + '.csv'), index=False)
        dashboard_score = dashboard['score']
        conf['dashboard_score'] = dashboard_score
        print(f'\t\tDashboard score: {round(dashboard_score, 2)}')

        print('\tPlotting')
        for outlier_removal in [True, False]:
            conf['outlier'] = outlier_removal
            plot_path = os.path.join(conf['output_path'], conf['output_file_name'] + ('_outlier' if conf['outlier'] else '') + '.pdf')
            if not os.path.exists(plot_path):
                save_figure(dashboard['solutions'], conf)
        try:
            timing_df = timing_df.append({
                "dataset": conf["dataset"], 
                "optimization": conf['optimization_method'], 
                "diversification": conf['diversification_method'], 
                "timing": conf['diversification_duration_s'], 
                "cadence": conf['time'], 
                "score":dashboard['score']
                }, ignore_index=True)
            timing_df.to_csv(os.path.join(DIVERSIFICATION_RESULT_PATH, "timing.csv"), index=False)
        except:
            print("I think a diversification result was already present.")
main()
