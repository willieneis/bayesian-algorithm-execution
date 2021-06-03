"""
Base classes.
"""

from argparse import Namespace
import copy

from .misc_util import dict_to_namespace


class Base:
    """Simple base class."""

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for this model.
        verbose : bool
            If True, print description string.
        """
        self.verbose_init_arg = verbose
        self.set_params(params)
        self.print_pre = '     * '
        if self.params.verbose:
            self.print_init()

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.name = getattr(params, 'name', 'Base')
        self.params.verbose = getattr(params, 'verbose', self.verbose_init_arg)

    def print_init(self):
        """Print a description string when object created."""
        print('*[INFO] Initialized ' + str(self))

    def set_print_params(self):
        """Set self.print_params."""
        if not hasattr(self, 'print_params'):
            self.print_params = copy.deepcopy(self.params)

    def __str__(self):
        self.set_print_params()
        return f'{self.params.name} with params={self.print_params}'
