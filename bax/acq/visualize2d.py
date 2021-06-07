"""
Code for visualizing acquisition functions and optimization.
"""

from argparse import Namespace
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams

from ..util.base import Base
from ..util.misc_util import dict_to_namespace


class AcqViz2D(Base):
    """
    Class to visualize acquisition function optimization for two-dimensional x.
    """

    def __init__(self, params=None, fig_ax=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the AcqOptimizer.
        fig_ax : tuple
            Tuple containing the pair (fig, ax)
        verbose : bool
            If True, print description string.
        """
        super().__init__(params, verbose)

        if fig_ax is None:
            fig, ax = plt.subplots(figsize=self.params.figsize)
        else:
            fig = fig_ax[0]
            ax = fig_ax[1]

        self.fig = fig
        self.ax = ax
        self.h_list = []

    def set_params(self, params):
        """Set self.params, the parameters for the AcqOptimizer."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "AcqViz2D")
        self.params.figsize = getattr(params, "figsize", (6, 6))
        self.params.n_path_max = getattr(params, "n_path_max", None)
        self.params.xlabel = getattr(params, "xlabel", "x")
        self.params.ylabel = getattr(params, "ylabel", "y")
        self.params.lims = getattr(params, "lims", None)

    def plot_function_contour(self, f_vec, domain, n_levels=100, grid=0.025):
        """Make contour plot, given np.vectorize'd function f_vec(X, Y)."""

        x = np.arange(domain[0][0], domain[0][1], grid)
        y = np.arange(domain[1][0], domain[1][1], grid)
        X, Y = np.meshgrid(x, y)
        Z = f_vec(X, Y)

        # For filled contours
        #cs = self.ax.contourf(X, Y, Z, n_levels, cmap=cm.Greens_r)

        # For contour lines
        cs = self.ax.contour(X, Y, Z, n_levels, cmap=cm.Greens_r)

    def plot_output_samples(self, output_list):
        """
        Plot algorithm outputs. This method assumes an optimization algorithm that
        returns 2d locations.
        """

        # Optionally crop, given self.n_path_max
        output_list = self.reduce_samp_list(output_list)

        for output in output_list:
            h = self.ax.plot(
                output[0],
                output[1],
                "^",
                color="blue",
                #color="#1f77b4",
                markersize=5,
                label="$\{ \\tilde{o}_\mathcal{A}^j \} \sim  p(o_\mathcal{A} | \mathcal{D}_t)$",
            )
        self.h_list.append(h[0])
        return h

    def plot_model_data(self, data):
        """Plot data, assumed to have attributes x and y."""
        x_list = [xin[0] for xin in data.x]
        y_list = [xin[1] for xin in data.x]
        #h = self.ax.plot(
            #x_list, y_list, "o", color="black", markersize=8
        #)
        h = self.ax.plot(
            x_list,
            y_list,
            "o",
            #color="deeppink",
            color="black",
            label="$\mathcal{D}_t = \{x_i, y_i\}_{i=1}^t$",
            markersize=7,
        )
        self.h_list.append(h[0])
        return h

    def plot_expected_output(self, expected_output):
        """Plot expected output."""
        self.ax.plot(
            expected_output[0],
            expected_output[1],
            "*",
            color='black',
            markersize=36,
        )
        h = self.ax.plot(
            expected_output[0],
            expected_output[1],
            "*",
            color='deeppink',
            markersize=33,
            label="$O_\mathcal{A}(\mathrm{E}[f | \mathcal{D}_t])$"
        )
        self.h_list.append(h[0])
        return h

    def plot_optima(self, optima_list):
        """Plot optima of a function."""
        for optima in optima_list:
            h = self.ax.plot(
                optima[0],
                optima[1],
                "s",
                color="#F3C807",
                markersize=12,
                label="$O_\mathcal{A}(f*)$",
            )
        self.h_list.append(h[0])
        return h

    def make_legend(self, h_list=None):
        """Make the legend."""
        if h_list is None:
            h_list = self.h_list

        # For legend within axes
        bbta = None
        #loc = 1
        loc = "lower left"
        ncol = 1

        # For legend above axes
        #self.ax.set_position([0.1, 0.1, 0.85, 0.7])
        #bbta = (0.5, 1.24)
        #loc = "upper center"
        #ncol = 3

        # Draw legend
        lgd = self.ax.legend(handles=h_list, loc=loc, bbox_to_anchor=bbta, ncol=ncol)

    def reduce_samp_list(self, samp_list):
        """Optionally reduce list of samples, based on self.n_path_max."""
        if self.params.n_path_max:
            samp_list = samp_list[:self.params.n_path_max]

        return samp_list
