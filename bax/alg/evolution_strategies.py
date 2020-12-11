"""
Evolution strategies as algorithms for BAX.
"""

from argparse import Namespace
import copy
import numpy as np

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
        self.params.samp_str = getattr(params, "samp_str", "cma")
        self.params.opt_mode = getattr(params, "opt_mode", "min")
        self.params.n_dim = len(self.params.init_x)
        self.params.n_dim_es = self.params.n_dim if self.params.n_dim>1 else 2

    def run_algorithm_on_f(self, f):
        """
        Run the algorithm by sequentially querying function f. Return the execution path
        and output.
        """
        # set self.params.sampler and self.params.gen_list
        if self.params.samp_str == 'cma':
            self.params.sampler = CMAES(
                self.params.n_dim_es,
                popsize=self.params.n_population,
                weight_decay=0.0,
                sigma_init = 0.2,
            )
        elif self.params.samp_str == 'mut':
            self.params.sampler = SimpleMutator(
                n_pop=self.params.n_population,
                init_list=[self.params.init_x],
                opt_mode=self.params.opt_mode,
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
            return self.params.gen_list.pop(0)

        except IndexError:
            max_iter = self.params.n_population * self.params.n_generation
            if len(exe_path.x) < max_iter:
                if len(exe_path.y):
                    self.params.sampler.tell(exe_path.y[-self.params.n_population:])

                next_gen_list = self.params.sampler.ask()
                next_gen_list = self.convert_next_gen_list(next_gen_list)
                self.params.gen_list = copy.deepcopy(next_gen_list)
                return self.params.gen_list.pop(0)
            else:
                return None

    def convert_next_gen_list(self, next_gen_list):
        """Optionally convert format of next_gen_list."""
        ngl = next_gen_list
        if not isinstance(next_gen_list, list):
            # Assume numpy array
            ngl = ngl.tolist()
        if self.params.n_dim == 1:
            ngl = [[ng[0]] for ng in ngl]
        return ngl

    def get_output_from_exe_path(self, exe_path):
        """Given an execution path, return algorithm output."""
        if self.params.opt_mode == "min":
            opt_idx = np.argmin(exe_path.y)
        elif self.params.opt_mode == "max":
            opt_idx = np.argmax(exe_path.y)

        return exe_path.x[opt_idx]


class SimpleMutator:
    """Class to perform simple mutation in evolutionary strategies."""

    def __init__(self, n_pop, init_list, opt_mode="min"):
        """Initalize."""
        self.n_pop = n_pop
        self.gen_list = copy.deepcopy(init_list)
        self.mut_list = []
        self.opt_mode = opt_mode

    def ask(self):
        """Mutate self.gen_list and return n_pop mutations."""
        gen_idx = np.random.randint(len(self.gen_list), size=self.n_pop)
        new_mut_list = []
        for i in gen_idx:
            mut = self.mutate_single(self.gen_list[i])
            new_mut_list.append(mut)

        # Save mutations as self.mut_list and return these
        self.mut_list = new_mut_list
        return new_mut_list

    def mutate_single(self, vec):
        """Mutate a single vector (stored as a list)."""
        normal_scale = 0.5
        vec_mut = []
        for num in vec:
            num_mut = np.random.normal(loc=num, scale=normal_scale)
            vec_mut.append(num_mut)
        return vec_mut

    def tell(self, val_list):
        """Re-make self.gen_list, using val_list (associated with self.mut_list)."""
        keep_frac = 0.3

        # If minimizing, reverse val_list
        if self.opt_mode == "min":
            val_list = -1 * np.array(val_list)

        keep_idx = np.argsort(val_list)[int((1 - keep_frac) * len(val_list)):]
        new_gen_list = [self.mut_list[i] for i in keep_idx[::-1]]
        self.gen_list = new_gen_list
