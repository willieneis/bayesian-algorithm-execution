"""
Algorithms for BAX.
"""

from argparse import Namespace
import copy
import numpy as np

from ..models.exe_path import ExePath
from ..util.misc_util import dict_to_namespace


class LinearScan:
    """
    Algorithm that scans over a grid on a one dimensional domain, and as output returns
    function values at each point on grid.
    """

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the algorithm.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        if verbose:
            self.print_str()

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.name = getattr(params, 'name', 'LinearScan')
        default_x_path = [[x] for x in np.linspace(3.5, 20, 100)]
        self.params.x_path = getattr(params, 'x_path', default_x_path)

    def run_algorithm_on_f(self, f):
        """
        Run the algorithm by sequentially querying function f. Return the execution path
        and output.
        """
        # Initialize execution path
        exe_path = Namespace(x=[], y=[])

        # Step through algorithm
        x = self.get_next_x(exe_path)
        while x is not None:
            y = f(x)
            exe_path.x.append(x)
            exe_path.y.append(y)
            x = self.get_next_x(exe_path)

        # Compute output from exe_path, and return both
        output = self.get_output_from_exe_path(exe_path)
        return exe_path, output

    def get_next_x(self, exe_path):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        len_path = len(exe_path.x)
        if len_path == len(self.params.x_path):
            next_x = None
        else:
            next_x = self.params.x_path[len_path]
        return next_x

    def get_output_from_exe_path(self, exe_path):
        """Given an execution path, return algorithm output."""
        return exe_path.y

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        print_params = copy.deepcopy(self.params)
        delattr(print_params, 'x_path')
        return f'{self.params.name} with params={print_params}'


class LinearScanRandGap(LinearScan):
    """
    Algorithm that scans over a grid (with randomly drawn gap size between gridpoints)
    on a one dimensional domain, and as output returns function values at each point on
    grid.
    """

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the algorithm.
        verbose : bool
            If True, print description string.
        """
        super().__init__(params, verbose)
        self.params.x_path_orig = copy.deepcopy(self.params.x_path)

    def run_algorithm_on_f(self, f):
        """
        Run the algorithm by sequentially querying function f. Return the execution path
        and output.
        """
        x_path = copy.deepcopy(self.params.x_path_orig)
        n_grid = len(x_path)
        rand_factor = 0.2
        min_gap = np.ceil((1 - rand_factor) * n_grid)
        max_gap = np.ceil((1 + rand_factor) * n_grid)
        new_n_grid = np.random.randint(min_gap, max_gap)
        min_x_path = np.min(x_path)
        max_x_path = np.max(x_path)
        new_x_path = [[x] for x in np.linspace(min_x_path, max_x_path, new_n_grid)]
        self.params.x_path = new_x_path

        return super().run_algorithm_on_f(f)

class AverageOutputs:
    """
    Algorithm that computes the average of function outputs for a set of input points.
    """

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the algorithm.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        if verbose:
            self.print_str()

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.name = getattr(params, 'name', 'AverageOutputs')
        default_x_path = [[5.1], [5.3], [5.5], [20.1], [20.3], [20.5]]
        self.params.x_path = getattr(params, 'x_path', default_x_path)

    def run_algorithm_on_f(self, f):
        """
        Run the algorithm by sequentially querying function f. Return the execution path
        and output.
        """
        # Initialize execution path
        exe_path = Namespace(x=[], y=[])

        # Step through algorithm
        x = self.get_next_x(exe_path)
        while x is not None:
            y = f(x)
            exe_path.x.append(x)
            exe_path.y.append(y)
            x = self.get_next_x(exe_path)

        # Compute output from exe_path, and return both
        output = self.get_output_from_exe_path(exe_path)
        return exe_path, output

    def get_next_x(self, exe_path):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        len_path = len(exe_path.x)
        if len_path == len(self.params.x_path):
            next_x = None
        else:
            next_x = self.params.x_path[len_path]
        return next_x

    def get_output_from_exe_path(self, exe_path):
        """Given an execution path, return algorithm output."""
        return np.mean(exe_path.y)

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        print_params = copy.deepcopy(self.params)
        delattr(print_params, 'x_path')
        return f'{self.params.name} with params={print_params}'
