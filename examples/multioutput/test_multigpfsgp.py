"""
Testing MultiGpfsGp and MultiBaxAcqFunction classes
"""

from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from bax.models.gpfs_gp import MultiGpfsGp
from bax.acq.acquisition import MultiBaxAcqFunction
from bax.acq.acqoptimize import AcqOptimizer
from bax.alg.algorithms import Algorithm
from bax.util.misc_util import dict_to_namespace
from bax.util.domain_util import unif_random_sample_domain, project_to_domain
import neatplot


# Set plot settings
neatplot.set_style()
neatplot.update_rc('figure.dpi', 120)
neatplot.update_rc('text.usetex', False)


# Set random seed
seed = 11
np.random.seed(seed)
tf.random.set_seed(seed)

# Global variables
DEFAULT_F_IS_DIFF = True
LONG_PATH = False


class NStep(Algorithm):
    """
    An algorithm that takes n steps through a state space (and touches n+1 states).
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, 'name', 'NStep')
        self.params.n = getattr(params, 'n', 10)
        self.params.f_is_diff = getattr(params, 'f_is_diff', DEFAULT_F_IS_DIFF)
        self.params.init_x = getattr(params, 'init_x', [0.0, 0.0])
        self.params.project_to_domain = getattr(params, 'project_to_domain', True)
        self.params.domain = getattr(params, 'domain', [[0.0, 10.0], [0.0, 10.0]])

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        len_path = len(self.exe_path.x)
        if len_path == 0:
            next_x = self.params.init_x
        elif len_path >= self.params.n + 1:
            next_x = None
        else:
            if self.params.f_is_diff:
                zip_path_end = zip(self.exe_path.x[-1], self.exe_path.y[-1])
                next_x = [xi + yi for xi, yi in zip_path_end]
            else:
                next_x = self.exe_path.y[-1]

            if self.params.project_to_domain:
                # Optionally, project to domain
                next_x = project_to_domain(next_x, self.params.domain)

        return next_x

    def get_output(self):
        """Return output based on self.exe_path."""
        return self.exe_path


def step_northwest(x_list, step_size=0.5, f_is_diff=DEFAULT_F_IS_DIFF):
    """Return x_list with a small positive value added to each element."""
    if f_is_diff:
        diffs_list = [step_size for x in x_list]
        return diffs_list
    else:
        x_list_new = [x + step_size for x in x_list]
        return x_list_new


def plot_path_2d(path, ax=None, true_path=False):
    """Plot a path through an assumed two-dimensional state space."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if true_path:
        ax.plot(x_plot, y_plot, 'k--', linewidth=3)
        ax.plot(x_plot, y_plot, '*', color='k', markersize=15)
    else:
        ax.plot(x_plot, y_plot, 'k--', linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, 'o', alpha=0.3)


# -------------
# Start Script
# -------------
# Set black-box function
f = step_northwest

# Set domain
domain = [[0, 23], [0, 23]] if LONG_PATH else [[0, 10], [0, 10]]

# Set algorithm
algo_class = NStep
n_steps = 40 if LONG_PATH else 15
algo_params = {'n': n_steps, 'init_x': [0.5, 0.5], 'domain': domain}
algo = algo_class(algo_params)

# Set model
gp_params = {'ls': 8.0, 'alpha': 5.0, 'sigma': 1e-2, 'n_dimx': 2}
multi_gp_params = {'n_dimy': 2, 'gp_params': gp_params}
gp_model_class = MultiGpfsGp

# Set data
data = Namespace()
n_init_data = 1
data.x = unif_random_sample_domain(domain, n_init_data)
data.y = [step_northwest(xi) for xi in data.x]

# Set acqfunction
acqfn_params = {'n_path': 30}
acqfn_class = MultiBaxAcqFunction
n_rand_acqopt = 1000

# Compute true path
true_algo = algo_class(algo_params)
true_path, _ = algo.run_algorithm_on_f(f)

# Run BAX loop
n_iter = 25

for i in range(n_iter):
    print('---' * 5 + f' Start iteration i={i} ' + '---' * 5)

    # Set model
    model = gp_model_class(multi_gp_params, data)

    # Set and optimize acquisition function
    acqfn = acqfn_class(acqfn_params, model, algo)
    x_test = unif_random_sample_domain(domain, n=n_rand_acqopt)
    acqopt = AcqOptimizer({"x_batch": x_test})
    x_next = acqopt.optimize(acqfn)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Plot observations
    x_obs = [xi[0] for xi in data.x]
    y_obs = [xi[1] for xi in data.x]
    ax.scatter(x_obs, y_obs, color='k', s=120)

    # Plot true path and posterior path samples
    plot_path_2d(true_path, ax, true_path=True)
    for path in acqfn.exe_path_list:
        plot_path_2d(path, ax)

    # Plot x_next
    ax.scatter(x_next[0], x_next[1], color='deeppink', s=120, zorder=100)

    # Plot settings
    ax.set(
        xlim=(domain[0][0], domain[0][1]),
        ylim=(domain[1][0], domain[1][1]),
        xlabel='$x_1$',
        ylabel='$x_2$',
    )

    save_figure = True
    if save_figure: neatplot.save_figure(f'bax_multi_{i}', 'pdf')

    # Query function, update data
    print(f'Length of data.x: {len(data.x)}')
    print(f'Length of data.y: {len(data.y)}')
    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)
