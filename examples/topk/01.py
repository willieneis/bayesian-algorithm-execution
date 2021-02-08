import copy
from argparse import Namespace, ArgumentParser
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf

from bax.alg.algorithms_new import TopK
from bax.models.simple_gp import SimpleGp
from bax.models.gpfs_gp import GpfsGp
from bax.models.stan_gp import get_stangp_hypers
from bax.acq.acquisition_new import BaxAcqFunction, RandBaxAcqFunction
from bax.acq.acqoptimize_new import AcqOptimizer
from bax.acq.visualize_new import AcqViz1D
from bax.util.domain_util import unif_random_sample_domain


# Parse args
parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--n_init", type=int, default=20)
parser.add_argument("--n_iter", type=int, default=70)
parser.add_argument(
    "--acq_str",
    type=str,
    default="eig3",
    choices=["eig1", "eig2", "eig3", "rand", "uncert"],
    help="Type of acquisition function. Choose one of eig1, eig2, eig3, rand, uncert.",
)
args = parser.parse_args()

# Set seeds
print(f"*[INFO] Seed: {args.seed}")
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# Set function
f_0 = lambda x: 2 * np.abs(x) * np.sin(x)
f = lambda x_list: np.sum([f_0(x) for x in x_list])

# Set algorithm  details
n_dim = 2
domain = [[-10, 10]] * n_dim
len_path = 200
k = 5
x_path = unif_random_sample_domain(domain, len_path)
algo = TopK({"x_path": x_path, "k": k})

# Get ground truth algorithm output
algo_gt = TopK({"x_path": x_path, "k": k, "name": "groundtruth"})
exepath_gt, output_gt = algo_gt.run_algorithm_on_f(f)
print(f"Algorithm ground truth output is:\n{output_gt}")

# Set metric
#metric = lambda x: algo.output_dist_fn_norm(x, output_gt)
metric = lambda x: algo.output_dist_fn_jaccard(x, output_gt)

# Set data for model
data = Namespace()
data.x = unif_random_sample_domain(domain, args.n_init)
data.y = [f(x) for x in data.x]

# Set model details
stan_hypers = get_stangp_hypers(f, n_samp=1000)
#gp_params = {"ls": 2.0, "alpha": 2.0, "sigma": 1e-2, "n_dimx": n_dim}
gp_params = {
    "ls": stan_hypers["ls"],
    "alpha": stan_hypers["alpha"],
    "sigma": stan_hypers["sigma"],
    "n_dimx": n_dim,
}
#modelclass = SimpleGp
modelclass = GpfsGp

# Set acquisition details
if args.acq_str == "eig1":
    acqfn_params = {"acq_str": "exe", "n_path": 100, "crop": False}    # EIG 1
    acq_cls = BaxAcqFunction
elif args.acq_str == "eig2":
    acqfn_params = {                                                   # EIG 2
        "acq_str": "out",
        "crop": False,
        "n_path": 1000,
        "min_neighbors": 3,
        "max_neighbors": 20,
        "dist_thresh": 1.0,
    }
    acq_cls = BaxAcqFunction
elif args.acq_str == "eig3":
    acqfn_params = {"acq_str": "exe", "n_path": 100, "crop": True}     # EIG 3
    acq_cls = BaxAcqFunction
elif args.acq_str == "rand":
    acq_cls = RandBaxAcqFunction
    acqfn_params = {}

# Set acqopt details
if args.acq_str == "eig2":
    n_acqopt = 1000
else:
    n_acqopt = 1500

# Set up results directory
results_dir = Path("examples/topk/results")
results_dir.mkdir(parents=True, exist_ok=True)

# Namespace to save results
results = Namespace(expected_metric_list = [])


# Run BAX loop
for i in range(args.n_iter):
    # Set model
    model = modelclass(gp_params, data)

    # Set and optimize acquisition function
    acqfn = acq_cls(acqfn_params, model, algo)
    x_test = unif_random_sample_domain(domain, n=n_acqopt)
    acqopt = AcqOptimizer({"x_batch": x_test})
    x_next = acqopt.optimize(acqfn)

    # Get expected metric
    metric_list = [metric(output) for output in acqfn.output_list]
    expected_metric = np.mean(metric_list)

    # Previous expected output
    #xl_list = []
    #for output in acqfn.output_list:
        #xl = []
        #list(map(xl.extend, output.x))
        #xl_list.append(xl)
    #expected_output = np.mean(xl_list, 0)

    # Print iter info
    print(f"Acqopt x_next = {x_next}")
    print(f"output_list[0] = {acqfn.output_list[0]}")
    print(f"expected_metric = {expected_metric}")
    print(f"Finished iter i = {i}")

    # Update results namespace
    results.expected_metric_list.append(expected_metric)

    # Query function, update data
    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)

# Save final data
results.data = data

# Pickle results
file_str = f"bax_{args.acq_str}_{args.seed}.pkl"
with open(results_dir / file_str, "wb") as handle:
    pickle.dump(results, handle)
    print(f"Saved results file: {results_dir}/{file_str}")
