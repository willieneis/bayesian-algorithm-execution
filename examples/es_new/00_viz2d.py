import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import tensorflow as tf

from bax.alg.evolution_strategies_new import EvolutionStrategies
from bax.models.simple_gp import SimpleGp
from bax.models.gpfs_gp import GpfsGp
from bax.models.stan_gp import get_stangp_hypers
from bax.acq.acquisition_new import BaxAcqFunction
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
#init_x = [6.0, 10.0] # Center-right start

domain = [[-5, 10], [0, 15]]
algo = EvolutionStrategies(
    {
        'n_generation': 15,
        'n_population': 8,
        'samp_str': 'mut',
        'opt_mode': 'min',
        'init_x': init_x,
        'domain': domain,
        'normal_scale': 0.5,
        'keep_frac': 0.3,
        'crop': False,
    }
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
acqfn_params = {"acq_str": "exe", "n_path": 100}
#acqfn_params = {
    #"acq_str": "out",
    #"n_path": 200,
    #"min_neighbors": 5,
    #"max_neighbors": 10,
    #"dist_thresh": 0.5,
#}

n_rand_acqopt = 350

# Run BAX loop
n_iter = 25

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
    h0 = vizzer.plot_output_samples(acqfn.output_list)
    h0b = vizzer.plot_expected_output(acqfn.output_list)
    h4 = vizzer.plot_model_data(model.data)
    h_list = [h4[0], h0[0], h0b[0]]
    vizzer.make_legend(h_list)

    #neatplot.save_figure(f"es_00_{i}")
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
