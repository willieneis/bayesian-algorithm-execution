import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from bax.alg.evolution_strategies import EvolutionStrategies
from bax.models.simple_gp import SimpleGp
from bax.models.gpfs_gp import GpfsGp
from bax.models.stan_gp import get_stangp_hypers
from bax.acq.acquisition import BaxAcqFunction
from bax.acq.acqoptimize import AcqOptimizer
from bax.util.domain_util import unif_random_sample_domain
from bax.acq.visualize2d import AcqViz2D

from branin import branin, branin_xy

import neatplot
neatplot.set_style('fonts')
neatplot.update_rc('font.size', 20)


seed = 11
np.random.seed(seed)
tf.random.set_seed(seed)


def run_algo_on_mean_f(model_mf, algo_mf, n_samp_mf):
    """Run algorithm on posterior mean (via MC estimate with n_samp samples)."""
    model_mf.initialize_function_sample_list(n_samp_mf)
    f_list = model_mf.call_function_sample_list
    f_mf = lambda x: np.mean(f_list([x for _ in range(n_samp_mf)]))
    exe_path_mf, output_mf = algo_mf.run_algorithm_on_f(f_mf)
    return exe_path_mf, output_mf


# Set function
f = branin

# Set algorithm details
init_x = [4.8, 13.0]
#init_x = [6.0, 10.0] # Center-right start

domain = [[-5, 10], [0, 15]]

algo_params = {
    'n_generation': 15,
    'n_population': 8,
    'samp_str': 'mut',
    'opt_mode': 'min',
    'init_x': init_x,
    'domain': domain,
    'normal_scale': 0.5,
    'keep_frac': 0.3,
    'crop': False,
    #'crop': True,
}
algo = EvolutionStrategies(algo_params)

# Set data for model
data = Namespace()
data.x = [init_x]
data.y = [f(x) for x in data.x]

# Set model details
gp_params = get_stangp_hypers(f, domain=domain, n_samp=200)
modelclass = GpfsGp

# Set acquisition details
acqfn_params = {"acq_str": "exe", "n_path": 100}

n_rand_acqopt = 350

# Run BAX loop
n_iter = 25

for i in range(n_iter):
    print('---' * 5 + f' Start iteration i={i} ' + '---' * 5)

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
    print(f"\tAcq optimizer x_next = {x_next}")
    print(f"\tCurrent expected_output = {expected_output}")
    print(f"\tCurrent output_mf = {output_mf}")
    print(f"\tCurrent f(expected_output) = {f(expected_output)}")
    print(f"\tCurrent f(output_mf) = {f(output_mf)}")
    expected_fout = np.mean([f(out) for out in acqfn.output_list])
    print(f"\tCurrent expected f(output) = {expected_fout}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    vizzer = AcqViz2D(fig_ax=(fig, ax))
    vizzer.plot_function_contour(branin_xy, domain)
    #h1 = vizzer.plot_output_samples(acqfn.output_list)
    h2 = vizzer.plot_model_data(model.data)
    h3 = vizzer.plot_expected_output(output_mf)
    h4 = vizzer.plot_optima([(-3.14, 12.275), (3.14, 2.275), (9.425, 2.475)])

    # Legend
    #vizzer.make_legend([h2[0], h4[0], h1[0], h3[0]]) # For out-of-plot legend
    #vizzer.make_legend([h2[0], h3[0], h1[0], h4[0]]) # For in-plot legend

    # Axis lims and labels
    offset = 0.3
    ax.set_xlim((domain[0][0] - offset, domain[0][1] + offset))
    ax.set_ylim((domain[1][0] - offset, domain[1][1] + offset))
    ax.set_aspect("equal", "box")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("InfoBAX")

    # Save plot
    #neatplot.save_figure(f"branin_bax_{i}", "pdf")

    # Show, pause, and close plot
    plt.show()
    inp = input("Press enter to continue (any other key to stop): ")
    if inp:
        break
    plt.close()

    # Query function, update data
    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)
