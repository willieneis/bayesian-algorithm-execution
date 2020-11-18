"""
Acquisition functions.
"""

from argparse import Namespace
import copy
import numpy as np

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
        self.params.acq_str = getattr(params, 'acq_str', 'pe')

    def entropy_given_normal_std(self, std_arr):
        """Return entropy given an array of 1D normal standard deviations."""
        entropy = np.log(std_arr) + np.log(np.sqrt(2 * np.pi)) + 0.5
        return entropy

    def acq_pe_normal(self, post_std, samp_std_list):
        """Predictive entropy acquisition function with normal inputs."""

        # Compute entropies for posterior predictive
        h_post = self.entropy_given_normal_std(post_std)

        # Compute entropies for posterior predictive given executin path samples
        h_samp_list = []
        for samp_std in samp_std_list:
            h_samp = self.entropy_given_normal_std(samp_std)
            h_samp_list.append(h_samp)

        avg_h_samp = np.mean(h_samp_list, 0)
        acq_pe = h_post - avg_h_samp
        return acq_pe

    def __call__(self, *args, **kwargs):
        """Class is callable and returns acquisition function on inputs."""
        if self.params.acq_str == 'pe':
            acq_function = self.acq_pe_normal

        return acq_function(*args, **kwargs)

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        print_params = copy.deepcopy(self.params)
        return f'{self.params.name} with params={print_params}'
