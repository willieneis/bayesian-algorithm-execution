"""
Classes for posterior function samples.
"""

from argparse import Namespace
import copy

from ..util.base import Base
from ..util.misc_util import dict_to_namespace


class FunctionSample(Base):
    """
    Posterior function sample, which takes a model as input.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the function sample."""
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "FunctionSample")

    def set_model(self, model):
        """Set self.model, self.data, and reset self.query_history."""
        self.model = copy.deepcopy(model)
        self.data = copy.deepcopy(self.model.data)
        self.reset_query_history()

    def reset_query_history(self):
        """Reset self.query_history."""
        self.query_history = Namespace(x=[], y=[])

    def set_query_history(self, query_history):
        """Set self.query_history to query_history."""
        self.query_history = query_history

    def set_all_obs(self):
        """Update self.all_obs given current self.model.data and self.query_history."""
        all_obs = Namespace()
        all_obs.x = self.data.x + self.query_history.x
        all_obs.y = self.data.y + self.query_history.y
        self.all_obs = all_obs

    def get_post_mean_std_list(self, x_list):
        """
        Return an array of posterior means and array of posterior std-devs (one element
        in array for each x in x_list).
        """

        # Set self.model.data using self.data and self.query_history
        self.set_all_obs()
        self.model.set_data(self.all_obs)

        # Compute posterior mean and std-dev
        mean_arr, std_arr = self.model.get_post_mu_cov(x_list, full_cov=False)
        return mean_arr, std_arr

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

    def __call__(self, x):
        """Class is callable and returns self.get_y(x)."""
        return self.get_y(x)
