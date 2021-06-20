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
    Evolution strategies for local minimization/maximization, computing a (local) optima
    in X as output, starting from some initial point.
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
        self.params.normal_scale = getattr(params, "normal_scale", 0.5)
        self.params.keep_frac = getattr(params, "keep_frac", 0.3)
        self.params.domain = getattr(params, "domain", [[0, 10]])
        self.params.n_dim = len(self.params.init_x)
        self.params.n_dim_es = self.params.n_dim if self.params.n_dim>1 else 2
        self.params.crop = getattr(params, "crop", True)

    def initialize(self):
        """Initialize algorithm, reset execution path."""
        super().initialize()

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
                normal_scale=self.params.normal_scale,
                keep_frac=self.params.keep_frac,
            )

        #self.params.gen_list = [self.params.init_x]
        self.params.gen_list = []   # TODO: figure out initialization

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        try:
            return self.params.gen_list.pop(0)

        except IndexError:
            max_iter = self.params.n_population * self.params.n_generation
            if len(self.exe_path.x) < max_iter:
                if len(self.exe_path.y):
                    self.params.sampler.tell(self.exe_path.y[-self.params.n_population:])

                next_gen_list = self.params.sampler.ask()
                next_gen_list = self.convert_next_gen_list(next_gen_list)
                next_gen_list = self.project_to_domain(next_gen_list)
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

    def project_to_domain(self, next_gen_list):
        """Project points in next_gen_list to be within self.params.domain."""
        next_gen_mat = np.array(next_gen_list)
        for col_idx in range(next_gen_mat.shape[1]):
            dom_ub = self.params.domain[col_idx][1]
            dom_lb = self.params.domain[col_idx][0]
            col_arr = next_gen_mat[:, col_idx]
            col_arr[col_arr > dom_ub] = dom_ub
            col_arr[col_arr < dom_lb] = dom_lb

        next_gen_list = next_gen_mat.tolist()
        return next_gen_list

    def get_exe_path_opt_idx(self):
        """Return the index of the optimal point in execution path."""
        if self.params.opt_mode == "min":
            opt_idx = np.argmin(self.exe_path.y)
        elif self.params.opt_mode == "max":
            opt_idx = np.argmax(self.exe_path.y)

        return opt_idx

    def get_output(self):
        """Given an execution path, return algorithm output."""
        opt_idx = self.get_exe_path_opt_idx()

        return self.exe_path.x[opt_idx]

    def get_exe_path_crop(self):
        """
        Return the minimal execution path for output, i.e. cropped execution path,
        specific to this algorithm.
        """
        opt_idx = self.get_exe_path_opt_idx()
        exe_path_crop = Namespace(x=[], y=[])

        if self.params.crop:
            exe_path_crop.x.append(self.exe_path.x[opt_idx])
            exe_path_crop.y.append(self.exe_path.y[opt_idx])
        else:
            exe_path_crop = self.exe_path

        return exe_path_crop


class SimpleMutator:
    """Class to perform simple mutation in evolutionary strategies."""

    def __init__(
        self, n_pop, init_list, opt_mode="min", normal_scale=0.5, keep_frac=0.3
    ):
        """Initalize."""
        self.n_pop = n_pop
        self.gen_list = copy.deepcopy(init_list)
        self.mut_list = []
        self.opt_mode = opt_mode
        self.normal_scale = normal_scale
        self.keep_frac = keep_frac

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
        vec_mut = []
        for num in vec:
            num_mut = np.random.normal(loc=num, scale=self.normal_scale)
            vec_mut.append(num_mut)
        return vec_mut

    def tell(self, val_list):
        """Re-make self.gen_list, using val_list (associated with self.mut_list)."""

        # If minimizing, reverse val_list
        if self.opt_mode == "min":
            val_list = -1 * np.array(val_list)

        keep_idx = np.argsort(val_list)[int((1 - self.keep_frac) * len(val_list)):]
        new_gen_list = [self.mut_list[i] for i in keep_idx[::-1]]
        self.gen_list = new_gen_list


class EvolutionStrategiesVal(EvolutionStrategies):
    """
    A version of EvolutionStrategies that returns the value of the optimal point.
    """

    def get_output(self):
        """Given an execution path, return algorithm output."""
        if self.params.opt_mode == "min":
            opt_val = np.min(self.exe_path.y)
        elif self.params.opt_mode == "max":
            opt_val = np.max(self.exe_path.y)

        return opt_val
