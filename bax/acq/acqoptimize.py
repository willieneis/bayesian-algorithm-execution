"""
Code for optimizing acquisition functions.
"""

from argparse import Namespace
import copy
import numpy as np
import matplotlib.pyplot as plt

from .acquisition import AcqFunction
from ..models.function import FunctionSample
from ..util.misc_util import dict_to_namespace
from ..util.timing import Timer


class AcqOptimizer:
    """
    Class for optimizing acquisition functions.
    """

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the AcqFunction.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        if verbose:
            self.print_str()

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.name = getattr(params, 'name', 'AcqOptimizer')
        self.params.opt_str = getattr(params, 'opt_str', 'rs')
        self.params.n_path = getattr(params, 'n_path', 20)
        default_x_test = [[x] for x in np.linspace(3.5, 40, 200)]
        self.params.x_test = getattr(params, 'x_test', default_x_test)
        self.params.viz_acq = getattr(params, 'viz_acq', True)

    def optimize(self, model, algo):
        """Optimize acquisition function."""
        # Optimize over fixed set x_test
        x_test = self.params.x_test

        # Set function sample with model
        fs = FunctionSample(verbose=False)
        fs.set_model(model)

        # Sample execution paths
        exe_path_list = []
        with Timer(f'Sample {self.params.n_path} execution paths'):
            for _ in range(self.params.n_path):
                exe_path_sample = self.sample_exe_path(fs, algo)
                exe_path_list.append(exe_path_sample)

        # Compute posterior, and post given each execution path, for x_test
        with Timer(f'Pre-compute acquisition at {len(x_test)} test points'):
            # Compute mean and std arrays for posterior
            mu, std = model.get_post_mu_cov(x_test, full_cov=False)

            # Compute mean and std arrays for posterior given execution path samples
            mu_list = []
            std_list = []
            for exe_path in exe_path_list:
                fs.set_query_history(exe_path)
                mu_samp, std_samp = fs.get_post_mean_std_list(x_test)
                mu_list.append(mu_samp)
                std_list.append(std_samp)

        # Compute acquisition function on x_test
        acqf = AcqFunction()
        acq_list = acqf(std, std_list)

        # Optionally: visualize acqoptimizer
        if self.params.viz_acq:
            self.viz_acqoptimizer(
                model, exe_path_list, acq_list, x_test, mu, std, mu_list, std_list
            )

        # Return optimizer of acquisition function
        acq_opt_idx = np.argmax(acq_list)
        acq_opt = x_test[acq_opt_idx]
        return acq_opt

    def sample_exe_path(self, fs, algo):
        """Return execution path sample given a function sample and algorithm."""
        fs.reset_query_history()
        exe_path, _ = algo.run_algorithm_on_f(fs)
        return exe_path

    def viz_acqoptimizer(
        self, model, exe_path_list, acq_list, x_test, mu, std, mu_list, std_list
    ):
        """
        Visualize the acquisition function, optimization, and related details, for a 1D
        continuous domain.
        """

        # Plot execution path samples
        for exe_path in exe_path_list:
            plt.plot(exe_path.x, exe_path.y, '.-', markersize=4, linewidth=0.5)

        # Plot posterior predictive
        lcb = mu - 3 * std
        ucb = mu + 3 * std
        plt.fill_between(np.array(x_test).reshape(-1), lcb, ucb, color='blue', alpha=0.1)

        # Plot posterior predictive given execution paths
        for mu_samp, std_samp in zip(mu_list, std_list):
            lcb = mu_samp - 3 * std_samp
            ucb = mu_samp + 3 * std_samp
            plt.fill_between(
                np.array(x_test).reshape(-1), lcb, ucb, color='blue', alpha=0.5
            )

        # Plot data
        plt.plot(model.data.x, model.data.y, 'o', color='deeppink')

        # Plot acquisition function
        ylim = plt.gca().get_ylim()
        acq_arr = np.array(acq_list)
        min_acq = np.min(acq_arr)
        max_acq = np.max(acq_arr)
        ylim_diff = ylim[1] - ylim[0]
        acq_height = 0.33 * ylim_diff
        ylim_new_min = ylim[0] - acq_height
        acq_arr = (acq_arr - min_acq) / (max_acq - min_acq) * acq_height + ylim_new_min
        plt.plot(np.array(x_test).reshape(-1), acq_arr, '-', color='red', linewidth=1)

        # Reset y axis
        plt.ylim([ylim_new_min, ylim[1]])

        # Plot dividing line
        xlim = plt.gca().get_xlim()
        plt.plot(xlim, [ylim[0], ylim[0]], '--', color='k')

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        print_params = copy.deepcopy(self.params)
        delattr(print_params, 'x_test')
        return f'{self.params.name} with params={print_params}'
