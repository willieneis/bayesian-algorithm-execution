"""
Classes for transforming data.
"""

from argparse import Namespace
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataTransformer:
    """Class for transforming data."""

    def __init__(self, data, verbose=True):
        """
        Parameters
        ----------
        data : Namespace
            Namespace containing data.
        verbose : bool
            If True, print description string.
        """
        self._set_y_data(data)
        self._set_y_transformers()
        self._set_verbose(verbose)

    def _set_y_data(self, data):
        """Set self.y_data."""

        # Ensure data.y is a numpy array
        data.y = np.array(data.y)

        if len(data.y.shape) > 1:
            if not (data.y.shape[0] == 1 or data.y.shape[1] == 1):
                raise ValueError('data.y has incorrect shape.')
        self.y_data_orig_shape = data.y.shape
        self.y_data = data.y.reshape(-1, 1)

    def _set_y_transformers(self):
        """Set transformers for self.y_data."""
        self.ss = StandardScaler()
        self.ss.fit(self.y_data)

    def _set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self._print_str()

    def transform_y_data(self, y_data=None):
        """Return transformed y_data (default self.y_data)."""

        # Set y_data and save y_data_orig_shape
        if y_data is None:
            y_data = self.y_data
            y_data_orig_shape = self.y_data_orig_shape
        else:
            y_data_orig_shape = y_data.shape

        # Transform y_data column
        y_data_col = y_data.reshape(-1, 1)
        y_trans_col = self.ss.transform(y_data_col)

        # Transform y_trans back to original shape
        y_trans = y_trans_col.reshape(y_data_orig_shape)
        return y_trans

    def inv_transform_y_data(self, y_data):
        """Return inverse transform of y_data."""
        y_data_orig_shape = y_data.shape

        # Inverse transform y_data column
        y_data_col = y_data.reshape(-1, 1)
        y_inv_trans_col = self.ss.inverse_transform(y_data_col)

        # Transform y_inv_trans back to original shape
        y_inv_trans = y_inv_trans_col.reshape(y_data_orig_shape)
        return y_inv_trans

    def _print_str(self):
        """Print a description string."""
        print('*DataTransformer')
