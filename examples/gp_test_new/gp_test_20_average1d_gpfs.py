import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from bax.alg.algorithms import AverageOutputs
from bax.models.gpfs_gp import GpfsGp
from bax.acq.acquisition_new import BaxAcqFunction
from bax.acq.acqoptimize_new import AcqOptimizer
from bax.acq.visualize import AcqViz1D

import neatplot
neatplot.set_style("fonts")


seed = 11
np.random.seed(seed)

# Set function
f = lambda x: 2 * np.sin(x[0])

# Set data for model
data = Namespace()
data.x = [[1.0]]
data.y = [f(x) for x in data.x]

# Set model as a GP
gp_params = {"ls": 2.0, "alpha": 2.0, "sigma": 1e-2}
model = GpfsGp(gp_params, data)

# Set arrays
min_x = 3.5
max_x = 20.0
len_path = 15
x_rand = np.random.uniform(min_x, max_x, len_path)
x_path = [[x] for x in x_rand]
x_test = [[x] for x in np.linspace(0.0, max_x, 500)]
y_test = [f(x) for x in x_test]

# Set algorithm
algo = AverageOutputs({"x_path": x_path})
algo_output_f = -0.9733


n_iter = 40

for i in range(n_iter):
    # Set and optimize acquisition function
    acqfn = BaxAcqFunction({"acq_str": "exe"}, model, algo)
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
    plt.ylim([-5.5, 6.5])
    plt.xlabel("x")
    plt.ylabel("y")

    vizzer = AcqViz1D()
    vizzer.plot_acqoptimizer_all(
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
    model = GpfsGp(gp_params, data)
