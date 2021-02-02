import copy
from argparse import Namespace, ArgumentParser
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf

from bax.alg.algorithms_new import GlobalOptUnifRandVal
from bax.models.gpfs_gp import GpfsGp
from bax.models.stan_gp import get_stangp_hypers
from bax.acq.acquisition_new import MesAcqFunction
from bax.acq.acqoptimize_new import AcqOptimizer
from bax.util.domain_util import unif_random_sample_domain

from ackley import Ackley

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
n_dim = 10
f = Ackley(n_dim)

# Set algorithm details
domain = [[-1, 1]] * n_dim
algo_params = {"opt_mode": "min", "domain": domain, "n_samp": 1500}
algo = GlobalOptUnifRandVal(algo_params)

# Set data for model
data = Namespace()
data.x = unif_random_sample_domain(domain, n=1)
data.y = [f(x) for x in data.x]

# Set model details
gp_params = get_stangp_hypers(f, domain=domain, n_samp=500)
modelclass = GpfsGp

# Set acquisition details
acqfn_params = {"opt_mode": "min", "n_path": 20}
n_rand_acqopt = 1500

# Set up results directory
results_dir = Path("examples/ackley/results")
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
n_iter = 200

for i in range(n_iter):
    # Set model
    model = modelclass(gp_params, data)

    # Set and optimize acquisition function
    acqfn = MesAcqFunction(acqfn_params, model, algo)
    x_test = unif_random_sample_domain(domain, n=n_rand_acqopt)
    acqopt = AcqOptimizer({"x_batch": x_test})
    x_next = acqopt.optimize(acqfn)

    # Compute current expected output
    output_xy_list = [
        exe_path.x[np.argmin(exe_path.y)] for exe_path in acqfn.exe_path_list
    ]
    expected_output = np.mean(output_xy_list, 0)

    # Compute output on mean function
    model_mf = modelclass(gp_params, data, verbose=False)
    algo_mf = GlobalOptUnifRandVal(algo_params, verbose=False)
    exe_path_mf, output_mf_val = run_algo_on_mean_f(model_mf, algo_mf, acqfn.params.n_path)
    output_mf = exe_path_mf.x[np.argmin(exe_path_mf.y)]

    # Print
    print(f"Acq optimizer x_next = {x_next}")
    print(f"Current expected_output = {expected_output}")
    print(f"Current output_mf = {output_mf}")
    print(f"Current f(expected_output) = {f(expected_output)}")
    print(f"Current f(output_mf) = {f(output_mf)}")
    expected_fout = np.mean([f(out) for out in output_xy_list])
    print(f"Current expected f(output) = {expected_fout}")
    print(f"Finished iter i = {i}")

    # Update results namespace
    results.expected_output_list.append(expected_output)
    results.f_expected_output_list.append(f(expected_output))
    results.output_mf_list.append(output_mf)
    results.f_output_mf_list.append(f(output_mf))
    results.expected_fout_list.append(expected_fout)

    # Query function, update data
    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)

# Save final data
results.data = data

# Pickle results
file_str = f"mes_{args.seed}.pkl"
with open(results_dir / file_str, "wb") as handle:
        pickle.dump(results, handle)
        print(f"Saved results file: {results_dir}/{file_str}")
