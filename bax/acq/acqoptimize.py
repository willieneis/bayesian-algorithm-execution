"""
Code for optimizing acquisition functions.
"""

from argparse import Namespace
import copy
import numpy as np

from .acquisition import AcqFunction
from .visualize import AcqViz1D, AcqViz2D
from ..models.function import FunctionSample
from ..util.misc_util import dict_to_namespace
from ..util.timing import Timer

import neatplot


class AcqOptimizer:
    """
    Class for optimizing acquisition functions.
    """

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the AcqOptimizer.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        if verbose:
            self.print_str()

    def set_params(self, params):
        """Set self.params, the parameters for the AcqOptimizer."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.name = getattr(params, "name", "AcqOptimizer")
        self.params.acq_str = getattr(params, "acq_str", "exe")
        self.params.opt_str = getattr(params, "opt_str", "rs")
        self.params.n_path = getattr(params, "n_path", 100)
        default_x_test = [[x] for x in np.linspace(0.0, 40.0, 500)]
        self.params.x_test = getattr(params, "x_test", default_x_test)
        self.params.viz_acq = getattr(params, "viz_acq", True)
        self.params.viz_dim = getattr(params, "viz_dim", 1)

    def optimize(self, model, algo):
        """Optimize acquisition function."""
        # Set function sample with model
        fs = FunctionSample(verbose=False)
        fs.set_model(model)

        # Sample execution paths and outputs
        exe_path_list, output_list = self.get_exe_path_and_output_samples(fs, algo)
        self.last_output_list = output_list

        # Define fixed set x_test over which we will optimize
        x_test = self.params.x_test

        # Compute acquisition function on each x in x_test
        acq_list, acq_vars = self.get_acqfunction_list(
            x_test, fs, model, exe_path_list, output_list
        )

        # Optionally visualize acqoptimizer
        self.viz_acqoptimizer(
            x_test, model, exe_path_list, output_list, acq_list, acq_vars
        )

        # Return optimizer of acquisition function
        acq_opt = x_test[np.argmax(acq_list)]
        return acq_opt

    def get_exe_path_and_output_samples(self, fs, algo):
        """
        Return exe_path_list and output_list respectively containing self.params.n_path
        exe path samples and associated outputs, given function sample fs and algo.
        """
        exe_path_list = []
        output_list = []
        with Timer(f"Sample {self.params.n_path} execution paths"):
            for _ in range(self.params.n_path):
                exe_path, output = self.sample_exe_path_and_output(fs, algo)
                exe_path_list.append(exe_path)
                output_list.append(output)
        return exe_path_list, output_list

    def sample_exe_path_and_output(self, fs, algo):
        """Return execution path sample given a function sample and algorithm."""
        fs.reset_query_history()
        exe_path, output = algo.run_algorithm_on_f(fs)
        return exe_path, output

    def sample_exe_path(self, fs, algo):
        """Return execution path sample given a function sample and algorithm."""
        # TODO: remove this and use above method (self.sample_exe_path_and_output) only?
        fs.reset_query_history()
        exe_path, _ = algo.run_algorithm_on_f(fs)
        return exe_path

    def sample_outputs(self, model, algo, n_path):
        """Return list of n_path output samples given model and algo."""
        # TODO: remove this and use above method (self.sample_exe_path_and_output) only?
        outputs = []
        fs = FunctionSample(verbose=False)
        fs.set_model(model)
        for _ in range(n_path):
            exe_path = self.sample_exe_path(fs, algo)
            output = algo.get_output_from_exe_path(exe_path)
            outputs.append(output)
        return outputs

    def post_mean_exe_path(self, fs, algo):
        """Return execution path of the posterior mean over functions"""
        fs.reset_query_history()
        # TODO: implement this.
        pass

    def get_acqfunction_list(self, x_test, fs, model, exe_path_list, output_list):
        """Compute acquisition function on each x in x_test."""
        with Timer(f"Compute acquisition function at {len(x_test)} test points"):
            # Compute posterior, and post given each execution path sample, for x_test
            mu, std = model.get_post_mu_cov(x_test, full_cov=False)

            # Compute mean and std arrays for posterior given execution path samples
            mu_list = []
            std_list = []
            for exe_path in exe_path_list:
                fs.set_query_history(exe_path)
                mu_samp, std_samp = fs.get_post_mean_std_list(x_test)
                mu_list.append(mu_samp)
                std_list.append(std_samp)

            # Compute acq_list, the acqfunction value for each x in x_test
            acqf = AcqFunction({"acq_str": self.params.acq_str})
            if self.params.acq_str == "exe":
                acq_list = acqf(std, std_list)
            elif self.params.acq_str == "out":
                acq_list = acqf(std, mu_list, std_list, output_list)

        # Package acq_vars
        acq_vars = {"mu": mu, "std": std, "mu_list": mu_list, "std_list": std_list}

        # Return list of acquisition function on x in x_test, and acq_vars
        return acq_list, acq_vars

    def viz_acqoptimizer(
        self, x_test, model, exe_path_list, output_list, acq_list, acq_vars
    ):
        """Optionally visualize acqoptimizer."""
        if self.params.viz_acq:
            with Timer(f"Visualize acquisition function"):
                if self.params.viz_dim == 1:
                    vizzer = AcqViz1D()
                elif self.params.viz_dim == 2:
                    vizzer = AcqViz2D()
                vizzer.plot_acqoptimizer_all(
                    model,
                    exe_path_list,
                    output_list,
                    acq_list,
                    x_test,
                    acq_vars["mu"],
                    acq_vars["std"],
                    acq_vars["mu_list"],
                    acq_vars["std_list"],
                )

    def get_last_output_list(self):
        """
        Return self.last_output_list, the last output_list computed during run of
        self.optimize().
        """
        return self.last_output_list

    def print_str(self):
        """Print a description string."""
        print("*[INFO] " + str(self))

    def set_print_params(self):
        """Set self.print_params."""
        self.print_params = copy.deepcopy(self.params)
        delattr(self.print_params, "x_test")

    def __str__(self):
        self.set_print_params()
        return f"{self.params.name} with params={self.print_params}"
