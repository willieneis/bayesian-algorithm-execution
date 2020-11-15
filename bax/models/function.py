"""
Classes for posterior function samples.
"""

from argparse import Namespace
import copy

from ..util.misc_util import dict_to_namespace


class FunctionSample:
    """
    Posterior function sample, which takes a model as input.
    """

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the function sample.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        if verbose:
            self.print_str()

    def set_params(self, params):
        """Set self.params, the parameters for the function sample."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.name = getattr(params, 'name', 'FunctionSample')

    def set_model(self, model):
        """Set self.model, self.data, and self.query_history."""
        self.model = copy.deepcopy(model)
        self.data = copy.deepcopy(self.model.data)
        self.reset_query_history()

    def reset_query_history(self):
        """Set self.query_history."""
        self.query_history = Namespace(x=[], y=[])

    def set_all_obs(self):
        """Update self.all_obs given current self.model.data and self.query_history."""
        all_obs = Namespace()
        all_obs.x = self.data.x + self.query_history.x
        all_obs.y = self.data.y + self.query_history.y
        self.all_obs = all_obs

    def get_y(self, x):
        """Sample and return output y at input x."""

        # Set self.model.data using self.data and self.query_history
        self.set_all_obs()
        self.model.set_data(self.all_obs)

        # Sample y from model posterior predictive
        y = self.model.sample_post_pred(x, 1)
        y = y[0]

        # Update self.query_history
        self.query_history.x.append(x)
        self.query_history.y.append(y)

        # Return output y
        return y

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        return f'{self.params.name} with params={self.params}'
