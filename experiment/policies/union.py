import functools
import numpy as np

from policies.policy import Policy
from algorithm import space as ALGORITHM_SPACE
from objective import objective_union
from functools import partial
from hyperopt import tpe, fmin, Trials

from utils.exhaustive_search import suggest


class Union(Policy):

    def run(self, X, y, input_space):
        super(Union, self).run(X, y)
        trials = Trials()
        space = {
            'pipeline': self.PIPELINE_SPACE,
            'algorithm': ALGORITHM_SPACE.get_domain_space(input_space, self.max_k),
        }
        obj_pl = functools.partial(objective_union,
                X=X,
                y=y,
                context=self.context,
                config=self.config)

        print('OPTIMIZATION:')
        print('#' * 50)
        fmin(
            fn=obj_pl,
            space=space,
            algo=partial(suggest, nbMaxSucessiveFailures=1000) if self.config['budget'] == 'inf' else tpe.suggest,
            max_evals=np.inf if self.config['budget'] == 'inf' else (None if self.config['budget_kind'] == 'time' else self.config['budget']),
            max_time=None if self.config['budget'] == 'inf' else (self.config['budget'] if self.config['budget_kind'] == 'time' else None),
            trials=trials,
            show_progressbar=False,
            verbose=0,
            rstate=np.random.RandomState(self.config['seed'])
        )
        print('#' * 50 + '\n')

        best_config = self.context['best_config']
        super(Union, self).display_step_results(best_config)