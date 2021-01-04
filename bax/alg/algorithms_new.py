"""
Algorithms for BAX.
"""

from argparse import Namespace
import copy
import numpy as np
from abc import ABC, abstractmethod

from ..util.misc_util import dict_to_namespace


class Algorithm(ABC):
    """Base class for a BAX Algorithm"""

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
        super().__init__()

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.name = getattr(params, "name", "Algorithm")

    def initialize(self):
        """Initialize algorithm, reset execution path."""
        self.exe_path = Namespace(x=[], y=[])

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        # Default behavior: return a uniform random value 10 times
        next_x = np.random.uniform() if len(self.exe_path.x) < 10 else None
        return next_x

    def take_step(self, f):
        """Take one step of the algorithm."""
        x = self.get_next_x()
        if x is not None:
            y = f(x)
            self.exe_path.x.append(x)
            self.exe_path.y.append(y)

        return x

    def run_algorithm_on_f(self, f):
        """
        Run the algorithm by sequentially querying function f. Return the execution path
        and output.
        """
        self.initialize()

        # Step through algorithm
        x = self.take_step(f)
        while x is not None:
            x = self.take_step(f)

        # Return execution path and output
        return self.exe_path, self.get_output()

    @abstractmethod
    def get_output(self):
        """Return output based on self.exe_path."""
        pass

    def print_str(self):
        """Print a description string."""
        print("*[INFO] " + str(self))

    def set_print_params(self):
        """Set self.print_params."""
        self.print_params = copy.deepcopy(self.params)

    def __str__(self):
        self.set_print_params()
        return f"{self.params.name} with params={self.print_params}"


class FixedPathAlgorithm(Algorithm):
    """
    Algorithm with a fixed execution path input sequence, specified by x_path parameter.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "FixedPathAlgorithm")
        self.params.x_path = getattr(params, "x_path", [])

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        len_path = len(self.exe_path.x)
        x_path = self.params.x_path
        next_x = x_path[len_path] if len_path < len(x_path) else None
        return next_x

    def get_output(self):
        """Return output based on self.exe_path."""
        # Default behavior: return execution path
        return self.exe_path

    def set_print_params(self):
        """Set self.print_params."""
        super().set_print_params()
        delattr(self.print_params, "x_path")


class LinearScan(FixedPathAlgorithm):
    """
    Algorithm that scans over a grid on a one dimensional domain, and as output returns
    function values at each point on grid.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "LinearScan")

    def get_output(self):
        """Return algorithm output given the execution path."""
        return self.exe_path.y


class LinearScanRandGap(LinearScan):
    """
    Algorithm that scans over a grid (with randomly drawn gap size between gridpoints)
    on a one dimensional domain, and as output returns function values at each point on
    grid.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "LinearScanRandGap")
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

    def set_print_params(self):
        """Set self.print_params."""
        super().set_print_params()
        delattr(self.print_params, "x_path_orig")


class AverageOutputs(FixedPathAlgorithm):
    """
    Algorithm that computes the average of function outputs for a set of input points.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "AverageOutputs")

    def get_output(self):
        """Return output based on self.exe_path."""
        return np.mean(self.exe_path.y)


class SortOutputs(FixedPathAlgorithm):
    """
    Algorithm that sorts function outputs for a set of input points.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "SortOutputs")

    def get_output(self):
        """Return output based on self.exe_path."""
        return np.argsort(self.exe_path.y)


class OptRightScan(Algorithm):
    """
    A simple minimization algorithm that, starting from some initial point, optimizes by
    scanning to the right until function starts to increase.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "OptRightScan")
        self.params.init_x = getattr(params, "init_x", [4.0])
        self.params.x_grid_gap = getattr(params, "x_grid_gap", 0.1)
        self.params.conv_thresh = getattr(params, "conv_thresh", 0.2)
        self.params.max_iter = getattr(params, "max_iter", 100)

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        len_path = len(self.exe_path.x)
        if len_path == 0:
            next_x = self.params.init_x
        else:
            next_x = [self.exe_path.x[-1][0] + self.params.x_grid_gap]

        if len_path >= 2:
            conv_max_val = np.min(self.exe_path.y[:-1]) + self.params.conv_thresh
            if self.exe_path.y[-1] > conv_max_val:
                next_x = None

        # Algorithm also has max number of steps
        if len_path > self.params.max_iter:
            next_x = None

        return next_x

    def get_output(self):
        """Return output based on self.exe_path."""
        return self.exe_path.x[-1]


class GlobalOptValGrid(FixedPathAlgorithm):
    """
    Algorithm that scans over a grid of points, and as output returns the minimum
    function value over the grid.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "GlobalOptGrid")
        self.params.opt_mode = getattr(params, "opt_mode", "min")

    def get_output(self):
        """Return output based on self.exe_path."""
        if self.params.opt_mode == "min":
            opt_val = np.min(self.exe_path.y)
        elif self.params.opt_mode == "max":
            opt_val = np.max(self.exe_path.y)

        return opt_val


class GlobalOptGrid(GlobalOptValGrid):
    """
    Algorithm that scans over a grid of points, and as output returns the minimum
    function input over the grid.
    """

    def get_output(self):
        """Return output based on self.exe_path."""
        if self.params.opt_mode == "min":
            opt_idx = np.argmin(self.exe_path.y)
        elif self.params.opt_mode == "max":
            opt_idx = np.argmax(self.exe_path.y)

        return self.exe_path.x[opt_idx]


class AlgorithmSet:
    """Wrapper that duplicates and manages a set of Algorithms."""

    def __init__(self, algo):
        """Set self.algo as an Algorithm."""
        self.algo = algo

    def run_algorithm_on_f_list(self, f_list, n_f):
        """
        Run the algorithm by sequentially querying f_list, which calls a list of n_f
        functions given an x_list of n_f inputs. Return the lists of execution paths and
        outputs.
        """
        algo_list = [copy.deepcopy(self.algo) for _ in range(n_f)]

        # Initialize each algo in list
        for algo in algo_list:
            algo.initialize()

        # Step through algorithms
        x_list = [algo.get_next_x() for algo in algo_list]
        while any(x is not None for x in x_list):
            y_list = f_list(x_list)
            x_list_new = []
            for algo, x, y in zip(algo_list, x_list, y_list):
                if x is not None:
                    algo.exe_path.x.append(x)
                    algo.exe_path.y.append(y)
                    x_next = algo.get_next_x()
                else:
                    x_next = None
                x_list_new.append(x_next)
            x_list = x_list_new

        # Collect exe_path_list and output_list
        exe_path_list = [algo.exe_path for algo in algo_list]
        output_list = [algo.get_output() for algo in algo_list]
        return exe_path_list, output_list

    def crop_exe_path(self, exe_path):
        """Return execution path without any Nones at end."""
        try:
            final_idx = next(i for i, x in enumerate(exe_path.x) if x==None)
        except StopIteration:
            final_idx = len(exe_path.x)

        del exe_path.x[final_idx:]
        del exe_path.y[final_idx:]
        return exe_path
