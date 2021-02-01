import copy
from argparse import Namespace, ArgumentParser
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf

from bax.alg.algorithms_new import GlobalOptUnifRandVal
from bax.models.gpfs_gp import GpfsGp
from bax.models.stan_gp import get_stangp_hypers
from bax.acq.acqoptimize_new import AcqOptimizer
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
tf.random.set_seed(args.seed)

# Set function
f = hartmann6

# Set algorithm details
n_dim = 6
domain = [[0, 1]] * n_dim
algo_params = {"opt_mode": "min", "domain": domain, "n_samp": 1000}
algo = GlobalOptUnifRandVal(algo_params)

# Set data for model
data = Namespace()
data.x = unif_random_sample_domain(domain, n=1)
data.y = [f(x) for x in data.x]

# Set model details
gp_params = get_stangp_hypers(f, domain=domain, n_samp=200)
modelclass = GpfsGp

# Set acquisition details
n_path = 20
n_rand_acqopt = 350

# Set up results directory
results_dir = Path("examples/hartmann/results")
results_dir.mkdir(parents=True, exist_ok=True)


# Useful function
def run_algo_on_mean_f(model_mf, algo_mf, n_samp_mf):
    """Run algorithm on posterior mean (via MC estimate with n_samp samples)."""
    model_mf.initialize_function_sample_list(n_samp_mf)
    f_list = model_mf.call_function_sample_list
    f_mf = lambda x: np.mean(f_list([x for _ in range(n_samp_mf)]))
    exe_path_mf, output_mf = algo_mf.run_algorithm_on_f(f_mf)
    return exe_path_mf, output_mf


# Namespace to save results
results = Namespace(
    expected_output_list = [],
    f_expected_output_list = [],
    output_mf_list = [],
    f_output_mf_list = [],
    expected_fout_list = [],
)

# Run BAX loop
n_iter = 80
x_all = unif_random_sample_domain(domain, n=n_iter)

for i in range(n_iter):
    # Set model
    model = modelclass(gp_params, data)

    # Set and optimize acquisition function
    x_next = x_all[i]

    # Compute output on mean function
    model_mf = modelclass(gp_params, data, verbose=False)
    algo_mf = GlobalOptUnifRandVal(algo_params, verbose=False)
    exe_path_mf, output_mf_val = run_algo_on_mean_f(model_mf, algo_mf, n_path)
    output_mf = exe_path_mf.x[np.argmin(exe_path_mf.y)]

    # Print
    print(f"Acq optimizer x_next = {x_next}")
    print(f"Current output_mf = {output_mf}")
    print(f"Current f(output_mf) = {f(output_mf)}")
    print(f"Finished iter i = {i}")

    # Update results namespace
    results.output_mf_list.append(output_mf)
    results.f_output_mf_list.append(f(output_mf))

    # Query function, update data
    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)

# Save final data
results.data = data

# Pickle results
file_str = f"rs_{args.seed}.pkl"
with open(results_dir / file_str, "wb") as handle:
        pickle.dump(results, handle)
        print(f"Saved results file: {results_dir}/{file_str}")
