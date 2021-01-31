"""
Hartmann6 synthetic benchmark function.
"""

from argparse import Namespace
import numpy as np


def hartmann6(x):
    """Hartmann6 synthetic function wrapper."""

    x = np.array(x)

    if len(x.shape) == 1:
        # x must be single input
        return hartmann6_single(x)

    elif len(x.shape) == 2:
        if x.shape[0] == 1 or x.shape[1] == 1:
            # x is single row matrix or single column matrix
            return hartmann6_single(x.reshape(-1))

        else:
            # For now, cannot handle multiple inputs
            raise ValueError('x has incorrect shape')

    else:
        raise ValueError('x has incorrect shape')


def hartmann6_single(x):
    """
    Hartmann6 synthetic function for an input x (np array shape (6, )).

    Optimal x* = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
    Minimal value f(x*) = -3.32237
    """

    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )

    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    log_sum_terms = (A * (P - x) ** 2).sum(axis=1)
    reward = alpha.dot(np.exp(-log_sum_terms))
    return -1 * reward
