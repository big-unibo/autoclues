import itertools

from hyperopt import hp
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion

from pipeline.space import *

def expand_params(operation, operator, space):
    expanded_params = {}

    if operator != None:

        try:
            params = globals()['params_{}'.format(type(operator).__name__)]()
        except Exception as e:
            params = {}

        updated_params = space[type(operator).__name__]
        params.update(updated_params)

        for param_name, param_val in params.items():
            expanded_params['{}__{}'.format(operation, param_name)] = param_val

    return expanded_params


def generate_grid(prototype, space):
    final_grid = []
    elements = [zip([k] * len(o), o) for k,o in prototype.items()]
    for element in itertools.product(*elements):
        config = dict(element)
        params = {}
        for operation, operator in config.items():
            config[operation] = [operator] # Trick since ScikitLearn requires list
            if operator is not None:
                params.update(expand_params(operation, operator, space))
        config.update(params)
        final_grid.append(config)
    return final_grid


def pretty_config(conf):
    print_conf = {}
    for k,v in conf.items():
        if '__' in k:
            print_conf[k] = v
        elif v == 'NoneType':
            print_conf[k] = None
        else:
            if isinstance(v, list):
                print_conf[k] = type(v[0]).__name__
            else:
                print_conf[k] = type(v).__name__
    return print_conf


def pretty_print_grid(grid):
    for conf in grid:
        print_conf = pretty_config(conf)
        print(print_conf)


def generate_domain_space(prototype, space):
    domain_space = {}
    print('SEARCH SPACE:')
    print('#' * 50)
    for operation, operators in prototype.items():
        print(f'\t{operation}')
        operators_space = []
        for operator in operators:
            print(f'\t\t{operator}')
            label = '{}_{}'.format(operation, type(operator).__name__ if operator is not None else 'NoneType')
            params = expand_params(operation, operator, space)
            operator_config = {}
            to_print_operator_config = ''
            for k, v in params.items():
                to_print_operator_config += '\t\t\t{}: {}\n'.format(k, v)
                operator_config[k] = hp.choice('{}_{}'.format(label, k), v)
            operators_space.append((label, operator_config))
            print(to_print_operator_config)
        domain_space[operation] = hp.choice(operation, operators_space)
    return domain_space

