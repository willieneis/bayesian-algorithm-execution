"""
Acquisition functions.
"""

from argparse import Namespace
import copy
import numpy as np
from scipy.stats import norm as sps_norm

from ..util.base import Base
from ..util.misc_util import dict_to_namespace
from ..util.timing import Timer
from ..models.function import FunctionSample
from ..alg.algorithms import AlgorithmSet


class AcqFunction(Base):
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
        super().__init__(params, verbose)
        self.set_model(model)

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)
        params = dict_to_namespace(params)
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


class RandAcqFunction(AcqFunction):
    """
    Class for random search acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'RandAcqFunction')

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""

        # Random acquisition function
        acq_list = [np.random.random() for x in x_list]

        return acq_list


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
        super().__init__(params, verbose)
        self.set_model(model)
        self.set_algorithm(algorithm)

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'AlgoAcqFunction')
        self.params.n_path = getattr(params, "n_path", 100)
        self.params.crop = getattr(params, "crop", True)

    def set_algorithm(self, algorithm):
        """Set self.algorithm for this acquisition function."""
        if not algorithm:
            raise ValueError("The algorithm input parameter cannot be None.")
        else:
            self.algorithm = algorithm.get_copy()

    def initialize(self):
        """
        Initialize the acquisition function before acquisition optimization. Draw
        samples of the execution path and algorithm output from functions sampled from
        the model.
        """
        exe_path_list, output_list, full_list = self.get_exe_path_and_output_samples()

        # Set self.output_list
        self.output_list = output_list
        self.exe_path_full_list = full_list

        # Set self.exe_path_list to list of full or cropped exe paths
        if self.params.crop:
            self.exe_path_list = exe_path_list
        else:
            self.exe_path_list = full_list

    def get_exe_path_and_output_samples_loop(self):
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

    def get_exe_path_and_output_samples(self):
        """
        Return exe_path_list and output_list respectively containing self.params.n_path
        exe_path samples and associated outputs, using self.model and self.algorithm.
        """
        exe_path_list = []
        output_list = []
        with Timer(f"Sample {self.params.n_path} execution paths"):
            # Initialize model fsl
            self.model.initialize_function_sample_list(self.params.n_path)

            # Run algorithm on function sample list
            f_list = self.model.call_function_sample_list
            algoset = AlgorithmSet(self.algorithm)
            exe_path_full_list, output_list = algoset.run_algorithm_on_f_list(
                f_list, self.params.n_path
            )

            # Get crop of each exe_path in exe_path_list
            exe_path_list = algoset.get_exe_path_list_crop()

        return exe_path_list, output_list, exe_path_full_list


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
        self.params.min_neighbors = getattr(params, "min_neighbors", 10)
        self.params.max_neighbors = getattr(params, "max_neighbors", 30)
        self.params.dist_thresh = getattr(params, "dist_thresh", 1.0)
        self.params.dist_thresh_init = getattr(params, "dist_thresh_init", 20.0)
        self.params.dist_thresh_inc = getattr(params, "dist_thresh_inc", 0.5)
        self.params.min_n_clust = getattr(params, "min_n_clust", 5)

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
        dist_thresh = self.params.dist_thresh_init
        cluster_idx_list = self.get_cluster_idx_list(output_list, dist_thresh)

        # -----
        print('\t- clust_idx_list initial details:')
        len_list = [len(clust) for clust in cluster_idx_list]
        print(f'\t\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}')
        # -----

        # Filter clusters that are too small (looping to find optimal dist_thresh)
        min_nn = self.params.min_neighbors
        min_n_clust = self.params.min_n_clust
        min_dist_thresh = self.params.dist_thresh

        cluster_idx_list_new = [clust for clust in cluster_idx_list if len(clust) > min_nn]
        # -----
        print('\t- clust_idx_list_NEW details:')
        len_list = [len(clust) for clust in cluster_idx_list_new]
        print(f'\t\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}')
        # -----
        while len(cluster_idx_list_new) > min_n_clust and dist_thresh >= min_dist_thresh:
            cluster_idx_list_keep = cluster_idx_list_new
            dist_thresh -= self.params.dist_thresh_inc
            print(f'NOTE: dist_thresh = {dist_thresh}')
            cluster_idx_tmp = self.get_cluster_idx_list(output_list, dist_thresh)
            cluster_idx_list_new = [clust for clust in cluster_idx_tmp if len(clust) > min_nn]

        try:
            cluster_idx_list = cluster_idx_list_keep
        except UnboundLocalError:
            print(
                'WARNING: cluster_idx_list_keep not assigned, using cluster_idx_list.'
            )
            pass

        ## Only remove small clusters if there are enough big clusters
        #if len(cluster_idx_list_new) > self.params.min_n_clust:
            #cluster_idx_list = cluster_idx_list_new

        # -----
        len_list = [len(clust) for clust in cluster_idx_list]
        print('\t- clust_idx_list final details:')
        print(f'\t\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}')
        print(f'\t\tFound dist_thresh: {dist_thresh}')
        # -----

        # Compute entropies for posterior predictive given execution path samples
        h_cluster_list = []
        std_cluster_list = []
        mean_cluster_list = []
        for idx_list in cluster_idx_list:
            # Mean of the mixture
            samp_mean_cluster_list = [samp_mean_list[idx] for idx in idx_list]
            samp_mean_cluster = np.mean(samp_mean_cluster_list, 0)
            mean_cluster_list.append(samp_mean_cluster)

            # Std of the mixture
            samp_std_cluster_list = [samp_std_list[idx] for idx in idx_list]
            smcls = [smc**2 for smc in samp_mean_cluster_list]
            sscls = [ssc**2 for ssc in samp_std_cluster_list]
            sum_smcls_sscls = [smcs + sscs for smcs, sscs in zip(smcls, sscls)]
            samp_sec_moment_cluster = np.mean(sum_smcls_sscls, 0)
            samp_var_cluster = samp_sec_moment_cluster - samp_mean_cluster**2
            samp_std_cluster = np.sqrt(samp_var_cluster)
            std_cluster_list.append(samp_std_cluster)

            # Entropy of the Gaussian approximation to the mixture
            h_cluster = self.entropy_given_normal_std(samp_std_cluster)
            h_cluster_list.extend([h_cluster])

        avg_h_cluster = np.mean(h_cluster_list, 0)
        acq_out = h_post - avg_h_cluster

        # Store variables
        self.cluster_idx_list = cluster_idx_list
        self.mean_cluster_list = mean_cluster_list
        self.std_cluster_list = std_cluster_list

        return acq_out

    def get_cluster_idx_list(self, output_list, dist_thresh):
        """
        Cluster outputs in output_list (based on nearest neighbors, and using
        dist_thresh as a nearness threshold) and return list of idx-list-per-cluster.
        """

        # Build distance matrix
        dist_fn = self.algorithm.get_output_dist_fn()
        dist_mat = [[dist_fn(o1, o2) for o1 in output_list] for o2 in output_list]

        # Build idx_arr_list and dist_arr_list
        idx_arr_list = []
        dist_arr_list = []
        for row in dist_mat:
            idx_sort = np.argsort(row)
            dist_sort = np.array([row[i] for i in idx_sort])

            # Keep at most max_neighbors, as long as they are within dist_thresh
            dist_sort = dist_sort[:self.params.max_neighbors]
            row_idx_keep = np.where(dist_sort < dist_thresh)[0]

            idx_arr = idx_sort[row_idx_keep]
            idx_arr_list.append(idx_arr)

            dist_arr = dist_sort[row_idx_keep]
            dist_arr_list.append(dist_arr)

        return idx_arr_list

    def acq_is_normal(
        self, post_std, samp_mean_list, samp_std_list, output_list, x_list
    ):
        """
        Algorithm-output-based acquisition function: EIG on the algorithm output, via
        the importance sampling strategy, for normal posterior predictive distributions.
        """
        # Compute list of means and stds for full execution path
        samp_mean_list_full = []
        samp_std_list_full = []
        for exe_path in self.exe_path_full_list:
            comb_data = Namespace()
            comb_data.x = self.model.data.x + exe_path.x
            comb_data.y = self.model.data.y + exe_path.y
            samp_mean, samp_std = self.model.gp_post_wrapper(
                x_list, comb_data, full_cov=False
            )
            samp_mean_list_full.append(samp_mean)
            samp_std_list_full.append(samp_std)

        # Compute entropies for posterior predictive
        h_post = self.entropy_given_normal_std(post_std)

        # Get list of idx-list-per-cluster
        cluster_idx_list = self.get_cluster_idx_list(
            output_list, self.params.dist_thresh
        )

        # -----
        print('\t- clust_idx_list details:')
        len_list = [len(clust) for clust in cluster_idx_list]
        print(f'\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}')
        # -----

        ## Remove clusters that are too small
        #min_nn = self.params.min_neighbors
        #cluster_idx_list = [clust for clust in cluster_idx_list if len(clust) > min_nn]

        # -----
        len_list = [len(clust) for clust in cluster_idx_list]
        print(f'\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}')
        # -----

        # Compute entropies for posterior predictive given execution path samples
        h_samp_list = []
        #for samp_mean, samp_std in zip(samp_mean_list, samp_std_list):
        for exe_idx in range(len(samp_mean_list)):
            # Unpack
            samp_mean = samp_mean_list[exe_idx]
            samp_mean_full = samp_mean_list_full[exe_idx]
            samp_std = samp_std_list[exe_idx]
            samp_std_full = samp_std_list_full[exe_idx]
            clust_idx = cluster_idx_list[exe_idx]

            # Sample from proposal distribution
            n_samp = 200
            pow_fac = 0.001
            samp_mat = np.random.normal(samp_mean, samp_std, (n_samp, len(samp_mean)))

            # Compute importance weights denominators
            mean_mat = np.vstack([samp_mean for _ in range(n_samp)])
            std_mat = np.vstack([samp_std for _ in range(n_samp)])
            pdf_mat = sps_norm.pdf(samp_mat, mean_mat, std_mat)
            #weight_mat_den = np.ones(pdf_mat.shape)
            weight_mat_den = pdf_mat

            # Compute importance weights numerators
            pdf_mat_sum = np.zeros(pdf_mat.shape)
            for idx in clust_idx:
                samp_mean_full = samp_mean_list_full[idx]
                samp_std_full = samp_std_list_full[idx]

                mean_mat = np.vstack([samp_mean_full for _ in range(n_samp)])
                std_mat = np.vstack([samp_std_full for _ in range(n_samp)])
                pdf_mat = sps_norm.pdf(samp_mat, mean_mat, std_mat)
                pdf_mat_sum = pdf_mat_sum + pdf_mat

            #weight_mat_num = np.ones(pdf_mat_sum.shape)
            weight_mat_num = pdf_mat_sum / len(clust_idx)
            weight_mat_num = weight_mat_num

            # Compute and normalize importance weights
            weight_mat = (weight_mat_num + 1e-50) / (weight_mat_den + 1.1e-50)
            weight_mat_norm = weight_mat / np.sum(weight_mat, 0)
            weight_mat_norm = (n_samp * weight_mat_norm) ** pow_fac

            # Reweight samples and compute statistics
            weight_samp = samp_mat * weight_mat_norm * n_samp
            is_mean = np.mean(weight_samp, 0)
            is_std = np.std(weight_samp, 0)
            h_samp = self.entropy_given_normal_std(is_std)
            h_samp_list.append(h_samp)

        avg_h_samp = np.mean(h_samp_list, 0)
        acq_is = h_post - avg_h_samp
        return acq_is

    def get_acq_list_batch(self, x_list):
        """Return acquisition function for a batch of inputs x_list."""

        with Timer(f"Compute acquisition function for a batch of {len(x_list)} points"):
            # Compute posterior, and post given each execution path sample, for x_list
            mu, std = self.model.get_post_mu_cov(x_list, full_cov=False)

            # Compute mean and std arrays for posterior given execution path samples
            mu_list = []
            std_list = []
            for exe_path in self.exe_path_list:
                comb_data = Namespace()
                comb_data.x = self.model.data.x + exe_path.x
                comb_data.y = self.model.data.y + exe_path.y
                samp_mu, samp_std = self.model.gp_post_wrapper(
                    x_list, comb_data, full_cov=False
                )
                mu_list.append(samp_mu)
                std_list.append(samp_std)

            # Compute acq_list, the acqfunction value for each x in x_list
            if self.params.acq_str == "exe":
                acq_list = self.acq_exe_normal(std, std_list)
            elif self.params.acq_str == 'out':
                acq_list = self.acq_out_normal(std, mu_list, std_list, self.output_list)
            elif self.params.acq_str == 'is':
                acq_list = self.acq_is_normal(
                    std, mu_list, std_list, self.output_list, x_list
                )

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
    Class for max-value entropy search acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'MesAcqFunction')
        self.params.opt_mode = getattr(params, "opt_mode", "max")

    def get_acq_list_batch(self, x_list):
        """Return acquisition function for a batch of inputs x_list."""

        with Timer(f"Compute acquisition function for a batch of {len(x_list)} points"):
            # Compute entropies for posterior for x in x_list
            mu, std = self.model.get_post_mu_cov(x_list, full_cov=False)
            h_post = self.entropy_given_normal_std(std)

            mc_list = []
            for output in self.output_list:
                if self.params.opt_mode == "max":
                    gam = (output - np.array(mu)) / np.array(std)
                elif self.params.opt_mode == "min":
                    gam = (np.array(mu) - output) / np.array(std)
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


class RandBaxAcqFunction(BaxAcqFunction):
    """
    Wrapper on BaxAcqFunction for random search acquisition, when we still want various
    BaxAcqFunction variables for visualizations.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'RandBaxAcqFunction')

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""
        acq_list = super().__call__(x_list) # NOTE: would super()(x_list) work?
        acq_list = [np.random.uniform() for _ in acq_list]
        return acq_list


class UsBaxAcqFunction(BaxAcqFunction):
    """
    Wrapper on BaxAcqFunction for uncertainty sampling acquisition, when we still want
    various BaxAcqFunction variables for visualizations.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'UsBaxAcqFunction')

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""
        super().__call__(x_list) # NOTE: would super()(x_list) work?
        acq_list = self.acq_vars["std"]
        return acq_list


class EigfBaxAcqFunction(BaxAcqFunction):
    """
    Wrapper on BaxAcqFunction for EIG-on-f acquisition function, when we still want
    various BaxAcqFunction variables for visualizations.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'EigfBaxAcqFunction')

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""
        super().__call__(x_list) # NOTE: would super()(x_list) work?
        std_list = self.acq_vars["std"]
        acq_list = self.entropy_given_normal_std(std_list)
        return acq_list


class MultiBaxAcqFunction(AlgoAcqFunction):
    """
    Class for computing BAX acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'MultiBaxAcqFunction')

    def entropy_given_normal_std(self, std_arr):
        """Return entropy given an array of 1D normal standard deviations."""
        entropy = np.log(std_arr) + np.log(np.sqrt(2 * np.pi)) + 0.5
        return entropy

    def entropy_given_normal_std_list(self, std_arr_list):
        """
        Return entropy given a list of arrays, where each is an array of 1D normal
        standard deviations.
        """
        entropy_list = [
            self.entropy_given_normal_std(std_arr) for std_arr in std_arr_list
        ]
        entropy = np.sum(entropy_list, 0)
        return entropy

    def acq_exe_normal(self, post_stds, samp_stds_list):
        """
        Execution-path-based acquisition function: EIG on the execution path, via
        predictive entropy, for normal posterior predictive distributions.
        """

        # Compute entropies for posterior predictive
        h_post = self.entropy_given_normal_std_list(post_stds)

        # Compute entropies for posterior predictive given execution path samples
        h_samp_list = []
        for samp_stds in samp_stds_list:
            h_samp = self.entropy_given_normal_std_list(samp_stds)
            h_samp_list.append(h_samp)

        avg_h_samp = np.mean(h_samp_list, 0)
        acq_exe = h_post - avg_h_samp
        return acq_exe

    def get_acq_list_batch(self, x_list):
        """Return acquisition function for a batch of inputs x_list."""

        # Compute posterior, and post given each execution path sample, for x_list
        with Timer(f"Compute acquisition function for a batch of {len(x_list)} points"):
            # NOTE: self.model is multimodel so the following returns a list of mus and
            # a list of stds
            mus, stds = self.model.get_post_mu_cov(x_list, full_cov=False)
            assert isinstance(mus, list)
            assert isinstance(stds, list)

            # Compute mean and std arrays for posterior given execution path samples
            mus_list = []
            stds_list = []
            for exe_path in self.exe_path_list:
                comb_data = Namespace()
                comb_data.x = self.model.data.x + exe_path.x
                comb_data.y = self.model.data.y + exe_path.y

                # NOTE: self.model is multimodel so the following returns a list of mus
                # and a list of stds
                samp_mus, samp_stds = self.model.gp_post_wrapper(
                    x_list, comb_data, full_cov=False
                )
                mus_list.append(samp_mus)
                stds_list.append(samp_stds)

            # Compute acq_list, the acqfunction value for each x in x_list
            acq_list = self.acq_exe_normal(stds, stds_list)

        # Package and store acq_vars
        self.acq_vars = {
            "mus": mus,
            "stds": stds,
            "mus_list": mus_list,
            "stds_list": stds_list,
            "acq_list": acq_list,
        }

        # Return list of acquisition function on x in x_list
        return acq_list

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""
        acq_list = self.get_acq_list_batch(x_list)
        return acq_list
