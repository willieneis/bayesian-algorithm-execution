import copy
from argparse import Namespace
import numpy as np

from bax.alg.evolution_strategies import EvolutionStrategies

from hartmann import hartmann6

import neatplot
neatplot.set_style('fonts')


seed = 11
np.random.seed(seed)


# Set function
f = hartmann6

# Set algorithm details
n_dim = 6

#init_x = unif_random_sample_domain(domain, n=1)
#init_x = [[0.0] * n_dim]
init_x = [[0.5] * n_dim]

domain = [[0, 1]] * n_dim

algo_params = {
    'n_generation': 50,
    'n_population': 10,
    'samp_str': 'mut',
    'opt_mode': 'min',
    'init_x': init_x[0],
    'domain': domain,
    'normal_scale': 0.05,
    'keep_frac': 0.3,
    'crop': False,
}
algo = EvolutionStrategies(algo_params)

# Run algorithm on f
exe_path, output = algo.run_algorithm_on_f(f)

# Print
min_idx = np.argmin(exe_path.y)
print(f'Best point so far x*: {exe_path.x[min_idx]}')
print(f'Value of best point so far f(x*): {exe_path.y[min_idx]}')
print(f'Found at iter: {min_idx}')
