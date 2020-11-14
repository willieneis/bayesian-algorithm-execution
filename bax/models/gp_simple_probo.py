"""
Classes for simple GP models without external PPL backend.
"""

from argparse import Namespace
import copy
import numpy as np

from .gp.gp_utils import kern_exp_quad, sample_mvn, gp_post
from ..util.data_transform import DataTransformer
from ..util.misc_util import dict_to_namespace


class SimpleGp:
    """
    Simple GP model without external PPL backend.
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
        self.params.trans_x = getattr(params, 'trans_x', False)
        self.params.trans_y = getattr(params, 'trans_y', False)

    def set_data(self, data):
        """Set self.data."""
        self.data_init = copy.deepcopy(data)
        self.data = copy.deepcopy(self.data_init)

        # Transform data.x
        self.data.x = self.transform_xin_list(self.data.x)

        # Transform data.y
        self.transform_data_y()

    def transform_xin_list(self, xin_list):
        """Transform list of xin (e.g. in data.x)."""
        # Ensure data.x is correct format (list of 1D numpy arrays)
        xin_list = [np.array(xin).reshape(-1) for xin in xin_list]

        if self.params.trans_x:
            # apply transformation to xin_list
            xin_list_trans = xin_list  # TODO: define default transformation
        else:
            xin_list_trans = xin_list

        return xin_list_trans

    def transform_data_y(self):
        """Transform data.y using DataTransformer."""
        if self.params.trans_y:
            self.dt = DataTransformer(self.data, False)
            y_trans = self.dt.transform_y_data()
        else:
            y_trans = self.data.y

        self.data = Namespace(x=self.data.x, y=y_trans)

    def inf(self, data):
        """Set data, run inference, update self.sample_list."""
        self.set_data(data)
        self.sample_list = [
            Namespace(
                ls=self.params.ls, alpha=self.params.alpha, sigma=self.params.sigma
            )
        ]

    def post(self, s):
        """Return one posterior sample."""
        return self.sample_list[0]

    def gen_list(self, x_list, z, s, nsamp):
        """
        Draw nsamp samples from generative process, given list of inputs
        x_list, posterior sample z, and seed s.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays each with shape=(-1,).
        z : Namespace
            Namespace of GP hyperparameters.
        s : int
            The seed, a positive integer.
        nsamp : int
            The number of samples to draw from generative process.

        Returns
        -------
        list
            A list with len=len(x_list) of numpy ndarrays, each with
            shape=(nsamp,).
        """
        x_list = self.transform_xin_list(x_list)
        pred_list = self.sample_gp_pred(nsamp, x_list, z)
        if self.params.trans_y:
            pred_list = [self.dt.inv_transform_y_data(pr) for pr in pred_list]
        return pred_list

    def postgen_list(self, x_list, s, nsamp):
        """
        Draw nsamp samples from posterior predictive distribution, given list
        of inputs x_list and seed s.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays each with shape=(-1,).
        s : int
            The seed, a positive integer.
        nsamp : int
            The number of samples to draw from the posterior predictive
            distribution.

        Returns
        -------
        list
            A list with len=len(x_list) of numpy ndarrays, each with
            shape=(nsamp,).
        """
        x_list = self.transform_xin_list(x_list)
        hp = self.sample_list[0]
        pred_list = self.sample_gp_post_pred(nsamp, x_list, hp, full_cov=True)
        if self.params.trans_y:
            pred_list = [self.dt.inv_transform_y_data(pr) for pr in pred_list]
        return pred_list

    def sample_gp_pred(self, nsamp, input_list, hp):
        """
        Sample from GP predictive distribution given one posterior GP sample.

        Parameters
        ----------
        nsamp : int
            Number of samples from predictive distribution.
        input_list : list
            A list of numpy ndarray shape=(-1, ).
        hp : Namespace
            Namespace of GP hyperparameters.

        Returns
        -------
        list
            A list of len=len(input_list) of numpy ndarrays shape=(nsamp, 1).
        """
        postmu, postcov = gp_post(
            self.data.x,
            self.data.y,
            input_list,
            hp.ls,
            hp.alpha,
            hp.sigma,
            self.params.kernel,
        )
        single_post_sample = sample_mvn(postmu, postcov, 1).reshape(-1)
        pred_list = [
            single_post_sample for _ in range(nsamp)
        ]  #### TODO: instead of duplicating this TS,
        #### sample nsamp times from generative
        #### process (given/conditioned-on this TS)
        return list(np.stack(pred_list).T)

    def sample_gp_post_pred(self, nsamp, input_list, hp, full_cov=False):
        """
        Sample from GP posterior predictive distribution.

        Parameters
        ----------
        nsamp : int
            Number of samples from posterior predictive distribution.
        input_list : list
            A list of numpy ndarray shape=(-1, ).
        hp : Namespace
            Namespace of GP hyperparameters.
        full_cov : bool
            If True, return covariance matrix, else return diagonal only.

        Returns
        -------
        list
            A list of len=len(input_list) of numpy ndarrays shape=(nsamp, 1).
        """
        postmu, postcov = gp_post(
            self.data.x,
            self.data.y,
            input_list,
            hp.ls,
            hp.alpha,
            hp.sigma,
            self.params.kernel,
            full_cov,
        )
        if full_cov:
            ppred_list = list(sample_mvn(postmu, postcov, nsamp))
        else:
            ppred_list = list(
                np.random.normal(
                    postmu.reshape(-1),
                    postcov.reshape(-1),
                    size=(nsamp, len(input_list)),
                )
            )

        return list(np.stack(ppred_list).T)

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        return f'SimpleGp with params={self.params}'
