"""
Code for optimizing acquisition functions.
"""

from argparse import Namespace
import copy
import numpy as np

from .acquisition_new import BaxAcqFunction # TODO: update
from ..util.misc_util import dict_to_namespace


class AcqOptimizer:
    """
    Class for optimizing acquisition functions.
    """

    def __init__(self, params=None, acqfunction=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the AcqOptimizer.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        if verbose:
            self.print_str()

    def set_params(self, params):
        """Set self.params, the parameters for the AcqOptimizer."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.name = getattr(params, "name", "AcqOptimizer")
        self.params.opt_str = getattr(params, "opt_str", "batch")
        default_x_batch = [[x] for x in np.linspace(0.0, 40.0, 500)]
        self.params.x_batch = getattr(params, "x_batch", default_x_batch)
        self.params.remove_x_dups = getattr(params, "remove_x_dups", False)

    def optimize(self, acqfunction):
        """
        Optimize acquisition function.

        Parameters
        ----------
        acqfunction : AcqFunction
            AcqFunction instance.
        """
        
        # Set self.acqfunction
        self.set_acqfunction(acqfunction)

        # Initialize acquisition function
        self.acqfunction.initialize()

        # Optimize acquisition function
        if self.params.opt_str == "batch":
            acq_opt = self.optimize_batch()

        return acq_opt

    def set_acqfunction(self, acqfunction):
        """Set self.acqfunction, the acquisition function."""
        if not acqfunction:
            # If acqfunction is None, set default acqfunction as BaxAcqFunction
            params = {"acq_str": "out"}
            self.acqfunction = BaxAcqFunction(params)
        else:
            self.acqfunction = acqfunction

    def optimize_batch(self):
        """Optimize acquisition function over self.params.x_batch."""
        x_batch = copy.deepcopy(self.params.x_batch)

        # Optionally remove data.x (in acqfunction) duplicates
        if self.params.remove_x_dups:
            x_batch = self.remove_x_dups(x_batch)

        # Optimize self.acqfunction over x_batch
        acq_list = self.acqfunction(x_batch)
        acq_opt = x_batch[np.argmax(acq_list)]

        return acq_opt

    def remove_x_dups(self, x_batch):
        """Remove elements of x_batch that are also in data.x (in self.acqfunction)"""

        # NOTE this requires self.acqfunction with model.data
        data = self.acqfunction.model.data

        # NOTE this only works for data.x consisting of list-types, not for arbitrary data.x
        for x in data.x:
            while True:
                try:
                    idx, pos = next(
                        (tup for tup in enumerate(x_batch) if all(tup[1] == x))
                    )
                    del x_batch[idx]
                except:
                    break

        return x_batch

    def print_str(self):
        """Print a description string."""
        print("*[INFO] " + str(self))

    def set_print_params(self):
        """Set self.print_params."""
        self.print_params = copy.deepcopy(self.params)
        delattr(self.print_params, "x_batch")

    def __str__(self):
        self.set_print_params()
        return f"{self.params.name} with params={self.print_params}"
