import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
#plt.ion()

from bax.alg.algorithms import GlobalOptValGrid
from bax.models.simple_gp import SimpleGp
from bax.acq.acquisition_new import MesAcqFunction
from bax.acq.acqoptimize_new import AcqOptimizer
from bax.acq.visualize import AcqViz1D

import neatplot
neatplot.set_style("fonts")


seed = 11
np.random.seed(seed)

# Set function
xm = 0.3
xa = 4.0
ym = 2.0
f = lambda x: ym * np.sin(np.pi * xm * (x[0] + xa)) + \
              ym * np.sin(2 * xm * np.pi * (x[0] + xa)) / 2.0

# Set data for model
data = Namespace()
data.x = [[4.0], [5.0], [7.3], [10.7], [11.8], [13.7], [15.4], [16.5], [17.6], [18.7]]
data.y = [f(x) for x in data.x]

# Set model as a GP
gp_params = {"ls": 1.0, "alpha": 2.0, "sigma": 1e-2}
model = SimpleGp(gp_params, data)

# Set min/max x
min_x = 3.5
max_x = 20.0

# Set x_path
n_path = 15
len_path = 75
x_path = [[x] for x in np.linspace(min_x, max_x, len_path)]

# Set x_test
n_test = 1000
x_test = [[x] for x in np.linspace(min_x, max_x, n_test)]
y_test = [f(x) for x in x_test]

# Set algorithm
algo = GlobalOptValGrid({"x_path": x_path, "opt_mode": "max"})

# Pre-computed algorithm output on f:
algo_output_f = 8.597

n_iter = 40

for i in range(n_iter):
    # Set and optimize acquisition function
    acqfn = MesAcqFunction({"n_path": n_path}, model, algo)
    acqopt = AcqOptimizer({"x_batch": x_test})
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
    fig = plt.figure(figsize=(8, 5))
    plt.xlim([0, max_x])
    plt.ylim([-7.0, 8.0])
    plt.xlabel("x")
    plt.ylabel("y")

    vizzer = AcqViz1D()
    h0 = vizzer.plot_exe_path_samples(acqfn.exe_path_list)
    h1 = vizzer.plot_postpred(x_test, acqfn.acq_vars["mu"], acqfn.acq_vars["std"])
    h3 = vizzer.plot_acqfunction(x_test, acqfn.acq_vars["acq_list"])
    h4 = vizzer.plot_model_data(model.data)
    h5 = vizzer.plot_acqoptima(acqfn.acq_vars["acq_list"], x_test)
    h_list = [h0[0], h4[0], h1, h5[0], h3[0]]
    vizzer.make_legend(h_list)
    plt.plot(x_test, y_test, "-", color="k", linewidth=2)

    #neatplot.save_figure(f"gp_test_14_{i}")
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

    # Update model
    model = SimpleGp(gp_params, data)
