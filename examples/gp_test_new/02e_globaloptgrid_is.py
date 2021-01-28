import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
#plt.ion()
import tensorflow as tf

from bax.alg.algorithms_new import GlobalOptGrid
from bax.models.simple_gp import SimpleGp
from bax.models.gpfs_gp import GpfsGp
from bax.models.stan_gp import get_stangp_hypers
from bax.acq.acquisition_new import BaxAcqFunction
from bax.acq.acqoptimize_new import AcqOptimizer
from bax.acq.visualize_new import AcqViz1D

import neatplot
neatplot.set_style("fonts")


seed = 11
np.random.seed(seed)
tf.random.set_seed(seed)

# Set function
xm = 0.3
xa = 4.0
ym = 2.0
f = lambda x: ym * np.sin(np.pi * xm * (x[0] + xa)) + \
              ym * np.sin(2 * xm * np.pi * (x[0] + xa)) / 2.0

# Set algorithm details
min_x = 3.5
max_x = 20.0
len_path = 200
x_path = [[x] for x in np.linspace(min_x, max_x, len_path)]
algo = GlobalOptGrid({"x_path": x_path, "opt_mode": "max"})

# Set data for model
data = Namespace()
data.x = [[4.0], [5.0], [7.3], [10.7], [11.8], [13.7], [15.4], [16.5], [17.6], [18.7]]
data.y = [f(x) for x in data.x]

# Set model details
gp_params = {"ls": 1.0, "alpha": 2.0, "sigma": 1e-2}
#gp_params = get_stangp_hypers(f, n_samp=200) # NOTE: can use StanGp to fit hypers
modelclass = GpfsGp
#modelclass = SimpleGp # NOTE: can use SimpleGp model

# Set acquisition details
acqfn_params = {
    "acq_str": "is",
    "n_path": 200,
    "min_neighbors": 0,
    "max_neighbors": 2,
    "dist_thresh": 10.0,
}
n_test = 500
x_test = [[x] for x in np.linspace(min_x, max_x, n_test)]
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
    output_list = acqfn.output_list
    output_list = [out[0] for out in output_list]
    expected_output = np.mean(output_list)

    # Print
    print(f"Acq optimizer x_next = {x_next}")
    print(f"Current expected_output = {expected_output}")
    print(f"Finished iter i = {i}")

    # Plot
    vizzer = AcqViz1D({"lims": (min_x, max_x, -7, 8), "n_path_max": 100})
    ax_tup = vizzer.plot_acqoptimizer_all(
        model,
        acqfn.exe_path_list,
        output_list,
        acqfn.acq_vars["acq_list"],
        x_test,
        acqfn.acq_vars["mu"],
        acqfn.acq_vars["std"],
        acqfn.acq_vars["mu_list"],
        acqfn.acq_vars["std_list"],
    )
    ax_tup[0].plot(x_test, y_test, "-", color="k", linewidth=2)

    #neatplot.save_figure(f"02_{i}")
    plt.show()

    # Pause
    inp = input("Press enter to continue (any other key to stop): ")
    if inp:
        break
    plt.close()
    del vizzer

    # Query function, update data
    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)
