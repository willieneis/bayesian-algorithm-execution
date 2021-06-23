"""
Utilities for domains (search spaces).
"""

import numpy as np


def unif_random_sample_domain(domain, n=1):
    """Draws a sample uniformly at random from domain (a list of tuple bounds)."""
    list_of_arr_per_dim = [np.random.uniform(dom[0], dom[1], n) for dom in domain]
    list_of_list_per_sample = [list(l) for l in np.array(list_of_arr_per_dim).T]
    return list_of_list_per_sample


def project_to_domain(x, domain):
    """Project x, a list of scalars, to be within domain (a list of tuple bounds)."""
    x_arr = np.array(x).reshape(-1)
    min_list = [tup[0] for tup in domain]
    max_list = [tup[1] for tup in domain]
    x_arr_clip = np.clip(x_arr, min_list, max_list)
    return list(x_arr_clip)
