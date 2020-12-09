"""
Acquisition functions.
"""

from argparse import Namespace
import copy
import numpy as np
from sklearn.cluster import KMeans

from ..util.misc_util import dict_to_namespace


class AcqFunction:
    """
    Class for computing acquisition functions.
    """

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the AcqFunction.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        if verbose:
            self.print_str()

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.name = getattr(params, 'name', 'AcqFunction')
        self.params.acq_str = getattr(params, 'acq_str', 'out')
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
        n_cluster_kmeans = np.min([self.params.n_cluster_kmeans, len(output_list)])
        km = KMeans(n_clusters=n_cluster_kmeans)
        km.fit(output_list)
        lab = km.labels_
        unq, unq_inv, unq_cnt = np.unique(lab, return_inverse=True, return_counts=True)
        idx_arr_list = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
        return idx_arr_list

    def __call__(self, *args, **kwargs):
        """Class is callable and returns acquisition function on inputs."""
        if self.params.acq_str == 'exe':
            acq_function = self.acq_exe_normal
        elif self.params.acq_str == 'out':
            acq_function = self.acq_out_normal

        return acq_function(*args, **kwargs)

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def set_print_params(self):
        """Set self.print_params."""
        self.print_params = copy.deepcopy(self.params)

    def __str__(self):
        self.set_print_params()
        return f'{self.params.name} with params={self.print_params}'
