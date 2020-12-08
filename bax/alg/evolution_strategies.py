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
        self.params.init_x = getattr(params, "init_x", [0.0])   # TODO: code currently requires init to 0
        self.params.n_population = getattr(params, "n_population", 5)
        self.params.n_generation = getattr(params, "n_generation", 3)
        self.params.n_dim = len(self.params.init_x)
        self.params.n_dim_es = self.params.n_dim if self.params.n_dim>1 else 2

    def run_algorithm_on_f(self, f):
        """
        Run the algorithm by sequentially querying function f. Return the execution path
        and output.
        """
        # set self.params.cmaes and self.params.gen_list
        self.params.cmaes = CMAES(
            self.params.n_dim_es,
            popsize=self.params.n_population,
            weight_decay=0.0,
            sigma_init = 0.2,
        )
        #self.params.gen_list = [self.params.init_x]
        self.params.gen_list = []   # TODO: figure out initialization

        # run algorithm in usual way on f
        return super().run_algorithm_on_f(f)

    def get_next_x(self, exe_path):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        try:
            return self.params.gen_list.pop()

        except IndexError:
            max_iter = self.params.n_population * self.params.n_generation
            if len(exe_path.x) < max_iter:
                if len(exe_path.y):
                    self.params.cmaes.tell(exe_path.y[-self.params.n_population:])

                next_gen_list = self.params.cmaes.ask()
                next_gen_list = self.convert_next_gen_list(next_gen_list)
                self.params.gen_list = next_gen_list
                return self.params.gen_list.pop()
            else:
                return None

    def convert_next_gen_list(self, next_gen_list):
        """Optionally convert format of next_gen_list."""
        ngl = next_gen_list.tolist()
        if self.params.n_dim == 1:
            ngl = [[ng[0]] for ng in ngl]
        return ngl

    def get_output_from_exe_path(self, exe_path):
        """Given an execution path, return algorithm output."""
        return exe_path.x[-1]
