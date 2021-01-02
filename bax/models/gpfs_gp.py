"""
Code for Gaussian processes using GPflow and GPflowSampling.
"""

from argparse import Namespace
import copy
import numpy as np
import tensorflow as tf
from gpflow import kernels

from .simple_gp import SimpleGp
from .gpfs.models import PathwiseGPR
from ..util.misc_util import dict_to_namespace, suppress_stdout_stderr
from ..util.domain_util import unif_random_sample_domain


class GpfsGp(SimpleGp):
    """
    GP model using GPFlowSampling.
    """

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.ls = getattr(params, 'ls', 3.7)
        self.params.sigma = getattr(params, 'sigma', 1e-2)
        self.params.name = getattr(params, 'name', 'GpfsGp')
        self.set_kernel(params)

    def set_kernel(self, params):
        """Return GPflow kernel."""
        self.params.kernel_str = getattr(params, 'kernel_str', 'rbf')
        ls = self.params.ls
        if self.params.kernel_str == 'rbf':
            self.params.kernel = kernels.SquaredExponential(lengthscales=ls)
        elif self.params.kernel_str == 'matern52':
            self.params.kernel = kernels.Matern52(lengthscales=ls)
        elif self.params.kernel_str == 'matern32':
            self.params.kernel = kernels.Matern32(lengthscales=ls)

    def set_data(self, data):
        """Set self.data."""
        super().set_data(data)
        self.tf_data = Namespace()
        self.tf_data.x = tf.convert_to_tensor(np.array(self.data.x))
        self.tf_data.y = tf.convert_to_tensor(
            np.array(self.data.y).reshape(-1, 1)
        )
        self.set_model()

    def set_model(self):
        """Set GPFlowSampling as self.model."""
        self.params.model = PathwiseGPR(
            data=(self.tf_data.x, self.tf_data.y),
            kernel=self.params.kernel,
            noise_variance=self.params.sigma**2,
        )
