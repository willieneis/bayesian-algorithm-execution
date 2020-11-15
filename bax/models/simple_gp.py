"""
Code for Gaussian processes.
"""

from argparse import Namespace
import copy
import numpy as np

from .gp.gp_utils import kern_exp_quad, sample_mvn, gp_post
from ..util.misc_util import dict_to_namespace


class SimpleGp:
    """
    Simple GP model without external backend.
    """

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for this model.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        if verbose:
            self.print_str()

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.ls = getattr(params, 'ls', 3.7)
        self.params.alpha = getattr(params, 'alpha', 1.85)
        self.params.sigma = getattr(params, 'sigma', 1e-5)
        self.params.kernel = getattr(params, 'kernel', kern_exp_quad)

        # Initialize self.data to be empty
        self.data = Namespace()
        self.data.x = []
        self.data.y = []

    def set_data(self, data):
        """Set self.data."""
        data = dict_to_namespace(data)
        self.data = copy.deepcopy(data)

    def get_gp_prior_mu_cov(self, x_list, full_cov=True):
        """
        Return GP prior parameters: mean (mu) and covariance (cov).

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays, each representing a domain point.
        full_cov : bool
            If True, return covariance matrix. If False, return list of standard
            deviations.

        Returns
        -------
        mu : ndarray
            A numpy 1d ndarray with len=len(x_list) of floats, corresponding to
            posterior mean for each x in x_list.
        cov : ndarray
            If full_cov is False, return a numpy 1d ndarray with len=len(x_list) of
            floats, corresponding to posterior standard deviations for each x in x_list.
            If full_cov is True, return the covariance matrix as a numpy ndarray
            (len(x_list) x len(x_list)).
        """
        # NOTE: currently assumes zero-mean prior.
        # TODO: generalized beyond zero-mean prior.
        mu = np.zeros(len(x_list))
        cov = self.params.kernel(x_list, x_list, self.params.ls, self.params.alpha)

        if full_cov is False:
            cov = np.sqrt(np.diag(cov))

        return mu, cov

    def get_gp_post_mu_cov(self, x_list, full_cov=True):
        """
        Return GP posterior parameters: mean (mu) and covariance (cov). If there is no
        data, return the GP prior parameters.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays, each representing a domain point.
        full_cov : bool
            If True, return covariance matrix. If False, return list of standard
            deviations.

        Returns
        -------
        mu : ndarray
            A numpy 1d ndarray with len=len(x_list) of floats, corresponding to
            posterior mean for each x in x_list.
        cov : ndarray
            If full_cov is False, return a numpy 1d ndarray with len=len(x_list) of
            floats, corresponding to posterior standard deviations for each x in x_list.
            If full_cov is True, return the covariance matrix as a numpy ndarray
            (len(x_list) x len(x_list)).
        """
        if len(self.data.x) == 0:
            return self.get_gp_prior_mu_cov(x_list, full_cov)

        # If data is not empty:

        mu, cov = gp_post(
            self.data.x,
            self.data.y,
            x_list,
            self.params.ls,
            self.params.alpha,
            self.params.sigma,
            self.params.kernel,
            full_cov=full_cov,
        )

        return mu, cov

    def get_gp_post_mu_cov_single(self, x):
        """Get GP posterior for an input x. Return posterior mean and std for x."""
        mu_arr, std_arr = self.get_gp_post_mu_cov([x], full_cov=False)
        return mu_arr[0], std_arr[0]

    def sample_prior_list(self, x_list, n_samp, full_cov=True):
        """Get samples from gp prior for each input in x_list."""
        mu, cov = self.get_gp_prior_mu_cov(x_list, full_cov)
        return self.get_normal_samples(mu, cov, n_samp, full_cov)

    def sample_prior(self, x, n_samp):
        """Get samples from gp prior for input x."""
        sample_list = self.sample_prior_list([x], n_samp)
        return sample_list[0]

    def sample_post_list(self, x_list, n_samp, full_cov=True):
        """Get samples from gp posterior for each input in x_list."""
        if len(self.data.x) == 0:
            return self.sample_prior_list(x_list, n_samp, full_cov)

        # If data is not empty:
        mu, cov = self.get_gp_post_mu_cov(x_list, full_cov)
        return self.get_normal_samples(mu, cov, n_samp, full_cov)

    def sample_post(self, x, n_samp):
        """Get samples from gp posterior for a single input x."""
        sample_list = self.sample_post_list([x], n_samp)
        return sample_list[0]

    def sample_post_pred_list(self, x_list, n_samp, full_cov=True):
        """Get samples from gp posterior predictive for each x in x_list."""
        # For now, return posterior (assuming low-noise case)
        # TODO: update this function
        return self.sample_post_list(x_list, n_samp, full_cov)

    def sample_post_pred(self, x, n_samp):
        """Get samples from gp posterior predictive for a single input x."""
        sample_list = self.sample_post_pred_list([x], n_samp)
        return sample_list[0]

    def get_normal_samples(self, mu, cov, n_samp, full_cov):
        """Return normal samples."""
        if full_cov:
            sample_list = list(sample_mvn(mu, cov, n_samp))
        else:
            sample_list = list(
                np.random.normal(
                    mu.reshape(-1,), cov.reshape(-1,), size=(n_samp, len(mu))
                )
            )
        x_list_sample_list = list(np.stack(sample_list).T)
        return x_list_sample_list

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        return f'SimpleGp with params={self.params}'
