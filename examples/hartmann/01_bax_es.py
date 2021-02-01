import copy
from argparse import Namespace, ArgumentParser
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf

from bax.alg.evolution_strategies_new import EvolutionStrategies
from bax.models.gpfs_gp import GpfsGp
from bax.models.stan_gp import get_stangp_hypers
from bax.acq.acquisition_new import BaxAcqFunction
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
init_x = unif_random_sample_domain(domain, n=1)
#init_x = [[0.0] *  n_dim]
#init_x = [[0.5] *  n_dim]

algo_params = {
    'n_generation': 50,
    'n_population': 10,
    'samp_str': 'mut',
    'opt_mode': 'min',
    'init_x': init_x[0],
    'domain': domain,
    'normal_scale': 0.05,
    'keep_frac': 0.3,
    #'crop': False,
    'crop': True,
}
#algo_params = {
    #'n_generation': 20,
    #'n_population': 10,
    #'samp_str': 'mut',
    #'opt_mode': 'min',
    #'init_x': init_x[0],
    #'domain': domain,
    #'normal_scale': 0.05,
    #'keep_frac': 0.3,
    #'crop': True,
#}
algo = EvolutionStrategies(algo_params)

# Set data for model
data = Namespace()
data.x = init_x
data.y = [f(x) for x in data.x]

# Set model details
gp_params = get_stangp_hypers(f, domain=domain, n_samp=500)
modelclass = GpfsGp

# Set acquisition details
acqfn_params = {"acq_str": "exe", "n_path": 20}
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

for i in range(n_iter):
    # Set model
    model = modelclass(gp_params, data)

    # Update algo.init_x
    algo.params.init_x = data.x[np.argmin(data.y)]

    # Set and optimize acquisition function
    acqfn = BaxAcqFunction(acqfn_params, model, algo)
    x_test = unif_random_sample_domain(domain, n=n_rand_acqopt)
    acqopt = AcqOptimizer({"x_batch": x_test})
    x_next = acqopt.optimize(acqfn)

    # Compute current expected output
    expected_output = np.mean(acqfn.output_list, 0)

    # Compute output on mean function
    model_mf = modelclass(gp_params, data, verbose=False)
    algo_mf = EvolutionStrategies(algo_params, verbose=False)
    algo_mf.params.init_x = data.x[np.argmin(data.y)]
    exe_path_mf, output_mf = run_algo_on_mean_f(model_mf, algo_mf, acqfn.params.n_path)

    # Print
    print(f"Acq optimizer x_next = {x_next}")
    print(f"Current expected_output = {expected_output}")
    print(f"Current output_mf = {output_mf}")
    print(f"Current f(expected_output) = {f(expected_output)}")
    print(f"Current f(output_mf) = {f(output_mf)}")
    expected_fout = np.mean([f(out) for out in acqfn.output_list])
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
file_str = f"bax_{args.seed}.pkl"
with open(results_dir / file_str, "wb") as handle:
    pickle.dump(results, handle)
    print(f"Saved results file: {results_dir}/{file_str}")
