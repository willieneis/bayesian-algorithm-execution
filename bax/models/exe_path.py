"""
Classes for execution paths.
"""

from argparse import Namespace
import copy

from ..util.base import Base
from ..util.misc_util import dict_to_namespace


class ExePath(Base):
    """
    Execution path, which takes a model as input.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the execution path."""
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'ExePath')

    def init_path_with_model(self, model):
        """Set self.model, self.exe_path, self.all_obs."""
        self.model = copy.deepcopy(model)
        self.data = copy.deepcopy(self.model.data)
        self.reset_exe_path()

    def reset_exe_path(self):
        """Set self.exe_path."""
        self.exe_path = Namespace(x=[], y=[])

    def set_all_obs(self):
        """Update self.all_obs given current self.model.data and self.exe_path."""
        all_obs = Namespace()
        all_obs.x = self.data.x + self.exe_path.x
        all_obs.y = self.data.y + self.exe_path.y
        self.all_obs = all_obs

    def next_step(self, x):
        """Take next step of execution path on input x, return output y."""
        self.set_all_obs()
        self.model.set_data(self.all_obs)
        y = self.model.sample_post_pred(x, 1)
        y = y[0]

        self.exe_path.x.append(x)
        self.exe_path.y.append(y)
        return y

    def get_y(self, x):
        """Call next_step method."""
        return self.next_step(x)
