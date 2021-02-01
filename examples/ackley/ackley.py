"""
Ackley benchmark function.
"""

import numpy as np
from scipy.stats import ortho_group


class Ackley:
    """
    Ackley function. Credits to David Eriksson for this.
    """

    def __init__(self, dim, active_dim=None, rotate=False):
        self.lb = -1 * np.ones(dim,)
        self.ub = 1 * np.ones(dim,)
        self.dim = dim
        self.active_dim = active_dim if active_dim else dim
        assert self.active_dim <= self.dim
        self.rotate = rotate
        self.q = ortho_group.rvs(self.dim) if rotate is True else np.eye(self.dim)

    def __call__(self, x):
        x = np.array(x)
        assert x.min() >= -1.0 and x.max() <= 1.0
        x = 0.5 * (x.copy() + 1)  # Map from [-1, 1] -> [0, 1]
        x = -5 + 15 * x  # Map from [0, 1] -> [-5, 10]
        assert len(x) == self.dim and np.all(x >= -5) and np.all(x <= 10)
        x = self.q[:, :self.active_dim].transpose() @ x

        val = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / self.active_dim)) - \
            np.exp(np.sum(np.cos(2.0 * np.pi * x)) / self.active_dim) + 20 + np.exp(1)
        return val
