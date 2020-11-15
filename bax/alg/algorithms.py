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
        and output."""
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
        output = self.get_output_from_exe_path(exe_path) #### TODO
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
