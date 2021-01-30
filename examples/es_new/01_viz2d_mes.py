import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import tensorflow as tf

#from bax.alg.evolution_strategies_new import EvolutionStrategiesVal
from bax.alg.algorithms_new import GlobalOptUnifRandVal
from bax.models.simple_gp import SimpleGp
from bax.models.gpfs_gp import GpfsGp
from bax.models.stan_gp import get_stangp_hypers
from bax.acq.acquisition_new import MesAcqFunction
from bax.acq.acqoptimize_new import AcqOptimizer
from bax.util.domain_util import unif_random_sample_domain
from bax.acq.visualize import AcqViz2D

from branin import branin

import neatplot
#neatplot.set_style('fonts')


seed = 11
np.random.seed(seed)
tf.random.set_seed(seed)

# Set function
f = branin

# Set algorithm details
init_x = [4.8, 13.0]
#init_x = [4.0, 14.0]
#init_x = [5.7, 13.25]
#init_x = [7.5, 13.0]
domain = [[-5, 10], [0, 15]]
algo = GlobalOptUnifRandVal(
    {"opt_mode": "min", "domain": domain, "n_samp": 300}
)

# Set data for model
data = Namespace()
data.x = [init_x]
data.y = [f(x) for x in data.x]

# Set model details
#gp_params = {"ls": 3.0, "alpha": 2.0, "sigma": 1e-2, "n_dimx": 2}
gp_params = get_stangp_hypers(f, domain=domain, n_samp=200) # NOTE: can use StanGp to fit hypers
modelclass = GpfsGp
#modelclass = SimpleGp # NOTE: can use SimpleGp model

# Set acquisition details
acqfn_params = {"opt_mode": "min", "n_path": 50}
n_rand_acqopt = 350

# Run BAX loop
n_iter = 25

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
    f_expected_output = f(expected_output)

    # Print
    print(f"Acq optimizer x_next = {x_next}")
    print(f"Current expected_output = {expected_output}")
    print(f"Current expected value f(expected_output) = {f_expected_output}")
    print(f"Finished iter i = {i}")

    # Plot
    fig = plt.figure(figsize=(6, 6))

    plt.xlim(domain[0])
    plt.ylim(domain[1])

    plt.xlabel('x')
    plt.ylabel('y')

    im = plt.imread('examples/es_new/branin_contour.png')
    implot = plt.imshow(
        im, extent=[domain[0][0], domain[0][1], domain[1][0], domain[1][1]]
    )

    vizzer = AcqViz2D()
    h0 = vizzer.plot_output_samples(output_xy_list)
    h0b = vizzer.plot_expected_output(output_xy_list)
    h4 = vizzer.plot_model_data(model.data)
    h_list = [h4[0], h0[0], h0b[0]]
    vizzer.make_legend(h_list)

    #neatplot.save_figure(f"es_01_{i}")
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
