import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
#plt.ion()
#import tensorflow as tf

from bax.alg.algorithms_new import AverageOutputs
from bax.models.simple_gp import SimpleGp
#from bax.models.gpfs_gp import GpfsGp
from bax.models.stan_gp import get_stangp_hypers
from bax.acq.acquisition_new import BaxAcqFunction
from bax.acq.acqoptimize_new import AcqOptimizer
from bax.acq.visualize_new import AcqViz1D

import neatplot
neatplot.set_style("fonts")
neatplot.update_rc("figure.dpi", 150)


seed = 11
np.random.seed(seed)
#tf.random.set_seed(seed)

# Set function
f = lambda x: 2 * np.sin(x[0])

# Set algorithm  details
min_x = 3.5
max_x = 20.0
len_path = 15
x_path = [[x] for x in np.random.uniform(min_x, max_x, len_path)]
algo = AverageOutputs({"x_path": x_path})
algo_output_f = -0.9733

# Set data for model
data = Namespace()
data.x = [[1.0]]
data.y = [f(x) for x in data.x]

# Set model details
gp_params = {"ls": 2.0, "alpha": 2.0, "sigma": 1e-2}
#gp_params = get_stangp_hypers(f, n_samp=200) # NOTE: can use StanGp to fit hypers
#modelclass = GpfsGp
modelclass = SimpleGp # NOTE: can use SimpleGp model

# Set acquisition details
acqfn_params = {"acq_str": "exe", "n_path": 100}
#acqfn_params = {        # NOTE: can use "out" acqfn
    #"acq_str": "out",
    #"crop": False,
    #"n_path": 200,
    #"min_neighbors": 10,
    #"max_neighbors": 20,
    #"dist_thresh": 0.05,
#}
x_test = [[x] for x in np.linspace(0.0, max_x, 500)]
y_test = [f(x) for x in x_test]
acqopt_params = {"x_batch": x_test}

# Run BAX loop
n_iter = 40

for i in range(n_iter):
    # Set model
    model = modelclass(gp_params, data)

    # Set and optimize acquisition function
    acqfn = BaxAcqFunction(acqfn_params, model, algo)
    acqopt = AcqOptimizer(acqopt_params)
    x_next = acqopt.optimize(acqfn)

    # Compute current expected output
    expected_output = np.mean(acqfn.output_list)
    out_abs_err = np.abs(algo_output_f - expected_output)

    # Print
    print(f"Acq optimizer x_next = {x_next}")
    print(f"Current expected_output = {expected_output}")
    print(f"Current output abs. error = {out_abs_err}")
    print(f"Finished iter i = {i}")

    # Plot
    vizzer = AcqViz1D({"lims": (0, max_x, -5.5, 6.5), "n_path_max": 50})
    ax_tup = vizzer.plot_acqoptimizer_all(
        model,
        acqfn.exe_path_list,
        acqfn.output_list,
        acqfn.acq_vars["acq_list"],
        x_test,
        acqfn.acq_vars["mu"],
        acqfn.acq_vars["std"],
        acqfn.acq_vars["mu_list"],
        acqfn.acq_vars["std_list"],
    )
    ax_tup[0].plot(x_test, y_test, "-", color="k", linewidth=2)

    #neatplot.save_figure(f"00b_{i}")
    plt.show()

    # Pause
    inp = input("Press enter to continue (any other key to stop): ")
    if inp:
        break
    plt.close()

    # Query function, update data
    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)
