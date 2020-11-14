"""
Miscellaneous utilities.
"""

from argparse import Namespace
import os


def dict_to_namespace(params):
    """
    If params is a dict, convert it to a Namespace, and return it.

    Parameters
    ----------
    params : Namespace_or_dict
        Namespace or dict.

    Returns
    -------
    params : Namespace
        Namespace of params
    """
    # If params is a dict, convert to Namespace
    if isinstance(params, dict):
        params = Namespace(**params)

    return params


class suppress_stdout_stderr:
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.

    Source: https://stackoverflow.com/q/11130156
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]

        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
