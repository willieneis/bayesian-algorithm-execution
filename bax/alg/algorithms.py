"""
Algorithms for BAX.
"""

from argparse import Namespace
import copy
import numpy as np
from abc import ABC, abstractmethod

from ..util.base import Base
from ..util.misc_util import dict_to_namespace
from ..util.domain_util import unif_random_sample_domain
from ..util.graph import jaccard_similarity


class Algorithm(ABC, Base):
    """Base class for a BAX Algorithm"""

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)
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

    def get_exe_path_crop(self):
        """
        Return the minimal execution path for output, i.e. cropped execution path,
        specific to this algorithm.
        """
        # As default, return untouched execution path
        return self.exe_path

    def get_copy(self):
        """Return a copy of this algorithm."""
        return copy.deepcopy(self)

    @abstractmethod
    def get_output(self):
        """Return output based on self.exe_path."""
        pass

    def get_output_dist_fn(self):
        """Return distance function for pairs of outputs."""

        # Default dist_fn casts outputs to arrays and returns Euclidean distance
        def dist_fn(a, b):
            a_arr = np.array(a)
            b_arr = np.array(b)
            return np.linalg.norm(a_arr - b_arr)

        return dist_fn


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
        self.params.crop_str = getattr(params, "crop_str", "min")

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

    def get_exe_path_crop(self):
        """
        Return the minimal execution path for output, i.e. cropped execution path,
        specific to this algorithm.
        """
        exe_path_crop = Namespace(x=[], y=[])
        min_idx = np.argmin(self.exe_path.y)

        if self.params.crop_str == "min":
            exe_path_crop.x.append(self.exe_path.x[min_idx])
            exe_path_crop.y.append(self.exe_path.y[min_idx])
        elif self.params.crop_str == "minplus":
            exe_path_crop.x.extend(self.exe_path.x[min_idx:])
            exe_path_crop.y.extend(self.exe_path.y[min_idx:])
        else:
            exe_path_crop = self.exe_path

        return exe_path_crop

    def get_output(self):
        """Return output based on self.exe_path."""
        min_idx = np.argmin(self.exe_path.y)
        min_pair = [self.exe_path.x[min_idx], self.exe_path.y[min_idx]]
        return min_pair

    def get_output_dist_fn(self):
        """Return distance function for pairs of outputs."""

        def dist_fn(a, b):
            a = np.array(a[0] + [a[1]])
            b = np.array(b[0] + [b[1]])
            return np.linalg.norm(a - b)

        return dist_fn


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

    def get_exe_path_opt_idx(self):
        """Return the index of the optimal point in execution path."""
        if self.params.opt_mode == "min":
            opt_idx = np.argmin(self.exe_path.y)
        elif self.params.opt_mode == "max":
            opt_idx = np.argmax(self.exe_path.y)

        return opt_idx

    def get_exe_path_crop(self):
        """
        Return the minimal execution path for output, i.e. cropped execution path,
        specific to this algorithm.
        """
        opt_idx = self.get_exe_path_opt_idx()

        exe_path_crop = Namespace(x=[], y=[])
        exe_path_crop.x.append(self.exe_path.x[opt_idx])
        exe_path_crop.y.append(self.exe_path.y[opt_idx])

        return exe_path_crop
        #return self.exe_path

    def get_output(self):
        """Return output based on self.exe_path."""
        opt_idx = self.get_exe_path_opt_idx()
        opt_val = self.exe_path.y[opt_idx]

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

        # Set opt_pair as [list, float]
        opt_pair = [self.exe_path.x[opt_idx], self.exe_path.y[opt_idx]]

        return opt_pair

    def get_output_dist_fn(self):
        """Return distance function for pairs of outputs."""

        def dist_fn(a, b):
            a = np.array(a[0] + [a[1]])
            b = np.array(b[0] + [b[1]])
            return np.linalg.norm(a - b)

        return dist_fn


class GlobalOptUnifRandVal(Algorithm):
    """
    Algorithm that performs global optimization over a domain via uniform random
    sampling and returns value of best input.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "GlobalOptUnifRandVal")
        self.params.opt_mode = getattr(params, "opt_mode", "min")
        self.params.domain = getattr(params, "domain", [[0, 10]])
        self.params.n_samp = getattr(params, "n_samp", 100)

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        if len(self.exe_path.x) < self.params.n_samp:
            # Return a random sample in domain
            next_x = unif_random_sample_domain(self.params.domain)[0]
        else:
            next_x = None

        return next_x

    def get_opt_idx(self):
        """Return index of optimal point in self.exe_path."""
        if self.params.opt_mode == "min":
            opt_idx = np.argmin(self.exe_path.y)
        elif self.params.opt_mode == "max":
            opt_idx = np.argmax(self.exe_path.y)

        return opt_idx

    def get_output(self):
        """Return output based on self.exe_path."""
        opt_idx = self.get_opt_idx()
        return self.exe_path.y[opt_idx]


class GlobalOptUnifRand(GlobalOptUnifRandVal):
    """
    Algorithm that performs global optimization over a domain via uniform random
    sampling and returns best input.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "GlobalOptUnifRand")

    def get_output(self):
        """Return output based on self.exe_path."""
        opt_idx = self.get_opt_idx()
        return self.exe_path.x[opt_idx]


class TopK(FixedPathAlgorithm):
    """
    Algorithm that scans over a set of points, and as output returns the K points with
    highest value.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "TopK")
        self.params.opt_mode = getattr(params, "opt_mode", "max")
        self.params.k = getattr(params, "k", 3)
        self.params.dist_str = getattr(params, "dist_str", "norm")

    def get_exe_path_topk_idx(self):
        """Return the index of the optimal point in execution path."""
        if self.params.opt_mode == "min":
            topk_idx = np.argsort(self.exe_path.y)[:self.params.k]
        elif self.params.opt_mode == "max":
            rev_exe_path_y = -np.array(self.exe_path.y)
            topk_idx = np.argsort(rev_exe_path_y)[:self.params.k]

        return topk_idx

    def get_exe_path_crop(self):
        """
        Return the minimal execution path for output, i.e. cropped execution path,
        specific to this algorithm.
        """
        topk_idx = self.get_exe_path_topk_idx()

        exe_path_crop = Namespace()
        exe_path_crop.x = [self.exe_path.x[idx] for idx in topk_idx]
        exe_path_crop.y = [self.exe_path.y[idx] for idx in topk_idx]

        return exe_path_crop

    def get_output(self):
        """Return output based on self.exe_path."""
        topk_idx = self.get_exe_path_topk_idx()

        out_ns = Namespace()
        out_ns.x = [self.exe_path.x[idx] for idx in topk_idx]
        out_ns.y = [self.exe_path.y[idx] for idx in topk_idx]

        return out_ns

    def get_output_dist_fn(self):
        """Return distance function for pairs of outputs."""
        if self.params.dist_str == "norm":
            dist_fn = self.output_dist_fn_norm
        elif self.params.dist_str == "jaccard":
            dist_fn = self.output_dist_fn_jaccard

        return dist_fn

    def output_dist_fn_norm(self, a, b):
        """Output dist_fn based on concatenated vector norm."""
        a_list = []
        list(map(a_list.extend, a.x))
        a_list.extend(a.y)
        a_arr = np.array(a_list)

        b_list = []
        list(map(b_list.extend, b.x))
        b_list.extend(b.y)
        b_arr = np.array(b_list)

        return np.linalg.norm(a_arr - b_arr)

    def output_dist_fn_jaccard(self, a, b):
        """Output dist_fn based on Jaccard similarity."""
        a_x_tup = [tuple(x) for x in a.x]
        b_x_tup = [tuple(x) for x in b.x]
        jac_sim = jaccard_similarity(a_x_tup, b_x_tup)
        dist = 1 - jac_sim
        return dist


class Noop(Algorithm):
    """"Dummy noop algorithm for debugging parallel code."""

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "Noop")

    def run_algorithm_on_f(self, f):
        """
        Run the dummy noop algorithm. Note that we must reseed each time or else child
        process will have the same state as parent, resulting in the same randomness.
        """
        np.random.seed()
        self.initialize()

        output = 0
        wait_time = random.randint(1, 5)
        rand = np.random.rand(1)
        print(f"Got {rand}. Going to wait for {wait_time} seconds")
        time.sleep(wait_time)
        print(f"Finished waiting for {wait_time} seconds")

        return self.exe_path, output

    def get_output(self):
        """Return output based on self.exe_path."""
        raise RuntimeError("Can't return output from execution path for Noop")


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
        algo_list = [self.algo.get_copy() for _ in range(n_f)]

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

        # Store algo_list
        self.algo_list = algo_list

        # Collect exe_path_list and output_list
        exe_path_list = [algo.exe_path for algo in algo_list]
        output_list = [algo.get_output() for algo in algo_list]
        return exe_path_list, output_list

    def get_exe_path_list_crop(self):
        """Return get_exe_path_crop for each algo in self.algo_list."""
        exe_path_list_crop = []
        for algo in self.algo_list:
            exe_path_crop = algo.get_exe_path_crop()
            exe_path_list_crop.append(exe_path_crop)

        return exe_path_list_crop

    def crop_exe_path_old(self, exe_path):
        """Return execution path without any Nones at end."""
        try:
            final_idx = next(i for i, x in enumerate(exe_path.x) if x==None)
        except StopIteration:
            final_idx = len(exe_path.x)

        del exe_path.x[final_idx:]
        del exe_path.y[final_idx:]
        return exe_path
