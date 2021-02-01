import copy
from argparse import Namespace, ArgumentParser
from pathlib import Path
import pickle
import numpy as np

from bax.alg.evolution_strategies import EvolutionStrategies
from bax.util.domain_util import unif_random_sample_domain

from hartmann import hartmann6

import neatplot
neatplot.set_style('fonts')


# Parse args
parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=11)
args = parser.parse_args()

# Set seeds
print(f"*[INFO] Seed: {args.seed}")
np.random.seed(args.seed)

# Set function
f = hartmann6

# Set algorithm details
n_dim = 6
domain = [[0, 1]] * n_dim
init_x = unif_random_sample_domain(domain, n=1)
#init_x = [[0.0] * n_dim]
#init_x = [[0.5] * n_dim]

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

# Set up results directory
results_dir = Path("examples/hartmann/results")
results_dir.mkdir(parents=True, exist_ok=True)

# Run algorithm on f
exe_path, output = algo.run_algorithm_on_f(f)

# Print
min_idx = np.argmin(exe_path.y)
print(f'Best point so far x*: {exe_path.x[min_idx]}')
print(f'Value of best point so far f(x*): {exe_path.y[min_idx]}')
print(f'Found at iter: {min_idx}')

# Namespace to save results
results = Namespace(
    expected_output_list = [],
    f_expected_output_list = [],
    output_mf_list = exe_path.x,
    f_output_mf_list = exe_path.y,
    expected_fout_list = [],
    data = exe_path,
)

# Pickle results
file_str = f"es_{args.seed}.pkl"
with open(results_dir / file_str, "wb") as handle:
        pickle.dump(results, handle)
        print(f"Saved results file: {results_dir}/{file_str}")
