"""
Code for Gaussian processes with hyperparameter fitting/sampling using Stan.
"""

from argparse import Namespace
import copy
import numpy as np

from .simple_gp import SimpleGp
from .stan.gp_fixedsig import get_stanmodel
from .gp.gp_utils import kern_exp_quad
from ..util.misc_util import dict_to_namespace, suppress_stdout_stderr
from ..util.domain_util import unif_random_sample_domain


class StanGp(SimpleGp):
    """
    GP model using Stan for hyperparameter fitting/sampling.
    """

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        # Set self.params
        self.params.name = getattr(params, 'name', 'StanGp')
        self.params.kernel = getattr(params, 'kernel', kern_exp_quad)
        self.params.ig1 = getattr(params, 'ig1', 4.0)
        self.params.ig2 = getattr(params, 'ig2', 3.0)
        self.params.n1 = getattr(params, 'n1', 1.0)
        self.params.n2 = getattr(params, 'n2', 1.0)
        self.params.n_iter = getattr(params, 'n_iter', 70)
        self.params.print_warnings = getattr(params, 'print_warnings', True)
        self.params.print_fit_result = getattr(params, 'print_fit_result', True)
        self.params.stanmodel = get_stanmodel(verbose=self.params.verbose)

    def fit_hypers(self):
        """Fit hyperparameters."""

        # Only fit hyperparameters if we have some observations
        if not self.data.x:
            raise Exception('self.data.x must not be empty in call to fit_hypers')

        data_dict = self.get_stan_data_dict()

        try:
            stanout = self.run_stan_optimizing(data_dict)
        except RuntimeError:
            if self.params.print_warnings:
                print(
                    '  > Stan LBFGS optimizer error. Running Newton optimizer instead.'
                )
            stanout = self.run_stan_optimizing(data_dict, stan_opt_str='Newton')

        self.set_hypers_from_stanout(stanout)
        if self.params.print_fit_result:
            self.print_fit_result()

    def get_stan_data_dict(self):
        """Return data_dict for input to stan."""
        n_dim_x = len(self.data.x[0])
        n_obs = len(self.data.x)
        data_dict = {
            'ig1': self.params.ig1,
            'ig2': self.params.ig2,
            'n1': self.params.n1,
            'n2': self.params.n2,
            'sigma': self.params.sigma,
            'D': n_dim_x,
            'N': n_obs,
            'x': self.data.x,
            'y': self.data.y,
        }
        return data_dict

    def run_stan_optimizing(self, data_dict, stan_opt_str='LBFGS', stanseed=543210):
        """Run self.stanmodel.optimizing and return output."""
        with suppress_stdout_stderr():
            stanout = self.params.stanmodel.optimizing(
                        data_dict,
                        iter=self.params.n_iter,
                        seed=stanseed,
                        as_vector=True,
                        algorithm=stan_opt_str,
                    )
        return stanout

    def set_hypers_from_stanout(self, stanout):
        """Set hyperparameters given a stanout object."""
        self.params.ls = float(stanout['rho'])
        self.params.alpha = float(stanout['alpha'])

    def print_fit_result(self):
        """Print result of hyparparameter fitting."""
        print('*[INFO] StanGp hyperparameter fitting:')
        print(f'  > ls pt est = {self.params.ls}')
        print(f'  > alpha pt est = {self.params.alpha}')
        print(f'  > sigma pt est = {self.params.sigma}')


def get_stangp_hypers(f, domain=[[0.0, 10.0]], n_samp=200):
    """Return hypers fit by StanGp, using n_samp random queries of function f."""

    # Construct dataset with n_samp unif random samples
    data = Namespace()
    data.x = unif_random_sample_domain(domain, n=n_samp)
    data.y = [f(x) for x in data.x]

    # Fit params with StanGp on data
    model = StanGp(data=data)
    model.fit_hypers()
    gp_hypers = {
        'ls': model.params.ls,
        'alpha': model.params.alpha,
        'sigma': model.params.sigma,
        'n_dimx': len(domain),
    }

    return gp_hypers
