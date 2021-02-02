import copy
from argparse import Namespace, ArgumentParser
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf

from bax.util.domain_util import unif_random_sample_domain

from hartmann import hartmann6


# Parse args
parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=11)
args = parser.parse_args()

# Set seeds
print(f"*[INFO] Seed: {args.seed}")
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# Set function
f = hartmann6

# Set algorithm details
n_dim = 6
domain = [[0, 1]] * n_dim

# Set data for model
data = Namespace()
data.x = unif_random_sample_domain(domain, n=1)
data.y = [f(x) for x in data.x]

# Set up results directory
results_dir = Path("examples/hartmann/results")
results_dir.mkdir(parents=True, exist_ok=True)


# Namespace to save results
results = Namespace(
    expected_output_list = [],
    f_expected_output_list = [],
    output_mf_list = [],
    f_output_mf_list = [],
    expected_fout_list = [],
)

# Run BAX loop
n_iter = 500
x_all = unif_random_sample_domain(domain, n=n_iter)

for i in range(n_iter):
    # Set and optimize acquisition function
    x_next = x_all[i]

    # Query function, update data
    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)

# Save final data
results.data = data

# Print
min_idx = np.argmin(data.y)
print(f'Best point so far x*: {data.x[min_idx]}')
print(f'Value of best point so far f(x*): {data.y[min_idx]}')
print(f'Found at iter: {min_idx}')

# Pickle results
file_str = f"rs2_{args.seed}.pkl"
with open(results_dir / file_str, "wb") as handle:
        pickle.dump(results, handle)
        print(f"Saved results file: {results_dir}/{file_str}")
