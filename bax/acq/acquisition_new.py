"""
Acquisition functions.
"""

from argparse import Namespace
import copy
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm as sps_norm

from ..util.misc_util import dict_to_namespace
from ..util.timing import Timer
from ..models.function import FunctionSample


class AcqFunction:
    """
    Class for computing acquisition functions.
    """

    def __init__(self, params=None, model=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the AcqFunction.
        model : SimpleGp
            Instance of a SimpleGp or child class.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        self.set_model(model)
        if verbose:
            self.print_str()

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.name = getattr(params, 'name', 'AcqFunction')

    def set_model(self, model):
        """Set self.model, the model underlying the acquisition function."""
        if not model:
            raise ValueError("The model input parameter cannot be None.")
        else:
            self.model = copy.deepcopy(model)

    def initialize(self):
        """Initialize the acquisition function before acquisition optimization."""
        pass

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""

        # Random acquisition function
        acq_list = [np.random.random() for x in x_list]

        return acq_list

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def set_print_params(self):
        """Set self.print_params."""
        self.print_params = copy.deepcopy(self.params)

    def __str__(self):
        self.set_print_params()
        return f'{self.params.name} with params={self.print_params}'


class AlgoAcqFunction(AcqFunction):
    """
    Class for computing acquisition functions involving algorithms, such as entropy
    search and BAX methods.
    """

    def __init__(self, params=None, model=None, algorithm=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the AcqFunction.
        model : SimpleGp
            Instance of a SimpleGp or child.
        algorithm : Algorithm
            Instance of an Algorithm or child.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        self.set_model(model)
        self.set_algorithm(algorithm)
        if verbose:
            self.print_str()

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'AlgoAcqFunction')
        self.params.n_path = getattr(params, "n_path", 100)

    def set_algorithm(self, algorithm):
        """Set self.algorithm for this acquisition function."""
        if not algorithm:
            raise ValueError("The algorithm input parameter cannot be None.")
        else:
            self.algorithm = copy.deepcopy(algorithm)

    def initialize(self):
        """
        Initialize the acquisition function before acquisition optimization. Draw
        samples of the execution path and algorithm output from functions sampled from
        the model.
        """
        exe_path_list, output_list = self.get_exe_path_and_output_samples()
        self.exe_path_list = exe_path_list
        self.output_list = output_list
        #self.last_output_list = output_list # TODO do I need this anymore?

    def get_exe_path_and_output_samples(self):
        """
        Return exe_path_list and output_list respectively containing self.params.n_path
        exe_path samples and associated outputs, using self.model and self.algorithm.
        """
        exe_path_list = []
        output_list = []
        with Timer(f"Sample {self.params.n_path} execution paths"):
            for _ in range(self.params.n_path):
                fs = FunctionSample(verbose=False)
                fs.set_model(self.model)
                exe_path, output = self.algorithm.run_algorithm_on_f(fs)
                exe_path_list.append(exe_path)
                output_list.append(output)

        return exe_path_list, output_list


class BaxAcqFunction(AlgoAcqFunction):
    """
    Class for computing BAX acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'BaxAcqFunction')
        self.params.acq_str = getattr(params, "acq_str", "exe")
        self.params.n_cluster_kmeans = getattr(params, 'n_cluster_kmeans', 35)

    def entropy_given_normal_std(self, std_arr):
        """Return entropy given an array of 1D normal standard deviations."""
        entropy = np.log(std_arr) + np.log(np.sqrt(2 * np.pi)) + 0.5
        return entropy

    def acq_exe_normal(self, post_std, samp_std_list):
        """
        Execution-path-based acquisition function: EIG on the execution path, via
        predictive entropy, for normal posterior predictive distributions.
        """

        # Compute entropies for posterior predictive
        h_post = self.entropy_given_normal_std(post_std)

        # Compute entropies for posterior predictive given execution path samples
        h_samp_list = []
        for samp_std in samp_std_list:
            h_samp = self.entropy_given_normal_std(samp_std)
            h_samp_list.append(h_samp)

        avg_h_samp = np.mean(h_samp_list, 0)
        acq_exe = h_post - avg_h_samp
        return acq_exe

    def acq_out_normal(self, post_std, samp_mean_list, samp_std_list, output_list):
        """
        Algorithm-output-based acquisition function: EIG on the algorithm output, via
        predictive entropy, for normal posterior predictive distributions.
        """

        # Compute entropies for posterior predictive
        h_post = self.entropy_given_normal_std(post_std)

        # Get list of idx-list-per-cluster
        samp_cluster_idx_list = self.get_samp_cluster_idx_list(output_list)

        # Compute entropies for posterior predictive given execution path samples
        h_samp_list = []
        for idx_list in samp_cluster_idx_list:
            # Mean of the mixture
            samp_mean_cluster_list = [samp_mean_list[idx] for idx in idx_list]
            samp_mean_cluster = np.mean(samp_mean_cluster_list, 0)

            # Std of the mixture
            samp_std_cluster_list = [samp_std_list[idx] for idx in idx_list]
            smcls = [smc**2 for smc in samp_mean_cluster_list]
            sscls = [ssc**2 for ssc in samp_std_cluster_list]
            sum_smcls_sscls = [smcs + sscs for smcs, sscs in zip(smcls, sscls)]
            samp_sec_moment_cluster = np.mean(sum_smcls_sscls, 0)
            samp_var_cluster = samp_sec_moment_cluster - samp_mean_cluster**2
            samp_std_cluster = np.sqrt(samp_var_cluster)

            # Entropy of the Gaussian approximation to the mixture
            h_samp = self.entropy_given_normal_std(samp_std_cluster)
            h_samp_list.extend([h_samp] * len(idx_list))

        avg_h_samp = np.mean(h_samp_list, 0)
        acq_out = h_post - avg_h_samp
        return acq_out

    def get_samp_cluster_idx_list(self, output_list):
        """
        Cluster outputs in output_list and return list of idx-list-per-cluster.
        TODO: implement algorithm-specific output clustering in another file or class.
        """
        if not isinstance(output_list[0], list):
            output_list = [[out] for out in output_list]
        n_cluster_kmeans = min(self.params.n_cluster_kmeans, len(output_list))
        km = KMeans(n_clusters=n_cluster_kmeans)
        km.fit(output_list)
        lab = km.labels_
        unq, unq_inv, unq_cnt = np.unique(lab, return_inverse=True, return_counts=True)
        idx_arr_list = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
        return idx_arr_list

    def get_acq_list_batch(self, x_list):
        """Return acquisition function for a batch of inputs x_list."""

        with Timer(f"Compute acquisition function for a batch of {len(x_list)} points"):
            # Compute posterior, and post given each execution path sample, for x_list
            mu, std = self.model.get_post_mu_cov(x_list, full_cov=False)

            # Compute mean and std arrays for posterior given execution path samples
            mu_list = []
            std_list = []
            for exe_path in self.exe_path_list:
                fs = FunctionSample(verbose=False)
                fs.set_model(self.model)
                fs.set_query_history(exe_path)

                mu_samp, std_samp = fs.get_post_mean_std_list(x_list)
                mu_list.append(mu_samp)
                std_list.append(std_samp)

            # Compute acq_list, the acqfunction value for each x in x_list
            if self.params.acq_str == "exe":
                acq_list = self.acq_exe_normal(std, std_list)
            elif self.params.acq_str == 'out':
                acq_list = self.acq_out_normal(std, mu_list, std_list, self.output_list)

        # Package and store acq_vars
        self.acq_vars = {
            "mu": mu,
            "std": std,
            "mu_list": mu_list,
            "std_list": std_list,
            "acq_list": acq_list,
        }

        # Return list of acquisition function on x in x_list
        return acq_list

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""
        acq_list = self.get_acq_list_batch(x_list)
        return acq_list


class MesAcqFunction(BaxAcqFunction):
    """
    Class for computing BAX acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'MesAcqFunction')

    def get_acq_list_batch(self, x_list):
        """Return acquisition function for a batch of inputs x_list."""

        with Timer(f"Compute acquisition function for a batch of {len(x_list)} points"):
            # Compute entropies for posterior for x in x_list
            mu, std = self.model.get_post_mu_cov(x_list, full_cov=False)
            h_post = self.entropy_given_normal_std(std)

            mc_list = []
            for output in self.output_list:
                gam = (output - np.array(mu)) / np.array(std)
                t1 = gam * sps_norm.pdf(gam) / (2 * sps_norm.cdf(gam))
                t2 = np.log(sps_norm.cdf(gam))
                mc_list.append(t1 - t2)
            acq_list = np.mean(mc_list, 0)

        # Package and store acq_vars
        self.acq_vars = {
            "mu": mu,
            "std": std,
            "acq_list": acq_list,
        }

        # Return list of acquisition function on x in x_list
        return acq_list
