"""
Evolution strategies as algorithms for BAX.
"""

from argparse import Namespace
import copy

from .algorithms import Algorithm
from ..estool.es import CMAES
from ..util.misc_util import dict_to_namespace


class EvolutionStrategies(Algorithm):
    """
    Evolution strategies for local minimization, computing a (local) optima in X as
    output, starting from some initial point.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "EvolutionStrategies")
        self.params.init_x = getattr(params, "init_x", [4.0])
        self.params.n_population = getattr(params, "n_population", 5)
        self.params.n_generation = getattr(params, "n_generation", 5)
        self.params.cmaes = CMAES(
            len(self.params.init_x),
            popsize=self.params.n_population,
            weight_decay=0.0,
            sigma_init = 0.5,
        )

    def get_next_x(self, exe_path):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        len_path = len(exe_path.x)
        if len_path == 0:
            next_x = self.params.init_x
        else:
            next_x = [exe_path.x[-1][0] + self.params.x_grid_gap]

        if len_path >= 2:
            conv_max_val = np.min(exe_path.y[:-1]) + self.params.conv_thresh
            if exe_path.y[-1] > conv_max_val:
                next_x = None

        # Algorithm also has max number of steps
        if len_path > self.params.max_iter:
            next_x = None

        return next_x

    def get_output_from_exe_path(self, exe_path):
        """Given an execution path, return algorithm output."""
        return exe_path.x[-1]
