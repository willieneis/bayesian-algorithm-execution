"""
Code for visualizing acquisition functions and optimization.
"""

from argparse import Namespace
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from ..util.base import Base
from ..util.misc_util import dict_to_namespace


class AcqViz1D(Base):
    """
    Class to visualize acquisition function optimization for one-dimensional x.
    """

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the AcqOptimizer.
        verbose : bool
            If True, print description string.
        """
        super().__init__(params, verbose)

        fig, ax = plt.subplots(figsize=self.params.figsize)
        self.fig = fig
        self.ax = ax
        self.h_list = []
        self.clist = rcParams['axes.prop_cycle']

    def set_params(self, params):
        """Set self.params, the parameters for the AcqOptimizer."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "AcqViz1D")
        self.params.figsize = getattr(params, "figsize", (8, 4))
        self.params.n_path_max = getattr(params, "n_path_max", None)
        self.params.xlabel = getattr(params, "xlabel", "x")
        self.params.ylabel = getattr(params, "ylabel", "y")
        self.params.lims = getattr(params, "lims", None)

    def plot_acqoptimizer_all(
        self,
        model,
        exe_path_list,
        output_list,
        acq_list,
        x_test,
        mu,
        std,
        mu_list,
        std_list,
    ):
        """
        Visualize the acquisition function, optimization, and related details, for a 1D
        continuous domain.
        """
        # Plot various details
        self.plot_postpred(x_test, mu, std)
        self.plot_acqfunction(x_test, acq_list)
        self.plot_exe_path_samples(exe_path_list)
        self.plot_model_data(model.data)
        self.plot_acqoptima(acq_list, x_test)
        self.plot_postpred_given_exe_path_samples(x_test, mu_list, std_list)
        self.plot_post_f_samples(model, x_test, exe_path_list)

        self.make_legend()
        self.set_post_plot_details()

        if getattr(self, "acqfunction_shown", False):
            ax_tup = (self.ax, self.ax_acq)
        else:
            ax_tup = (self.ax,)

        return ax_tup

    def plot_postpred(self, x_test, mu, std, noise=0.0):
        """Plot posterior predictive distribution."""
        std = std + noise
        lcb = mu - 3 * std
        ucb = mu + 3 * std
        h = self.ax.fill_between(
            np.array(x_test).reshape(-1),
            lcb,
            ucb,
            color=(1.0, 0.9255, 0.7961, 1.0),
            #color="orange",
            #alpha=0.18,
            label="$p(y|\mathcal{D}_t, x)$",
        )
        self.h_list.append(h)
        return h

    def plot_exe_path_crop_samples(self, exe_path_list):
        """Plot execution path samples."""

        # Optionally crop, given self.n_path_max
        exe_path_list = self.reduce_samp_list(exe_path_list)

        # Reset color cycle
        cgen = itertools.cycle(self.clist)

        for exe_path in exe_path_list:
            h = self.ax.plot(
                exe_path.x,
                exe_path.y,
                "x",
                #color=next(cgen)['color'],
                #color="#d62728",
                #color="deeppink",
                color="magenta",
                markersize=10,
                linewidth=0.1,
                label="$\{ \\tilde{O}_\mathcal{A}^j \} \sim p(O_\mathcal{A}(f) | \mathcal{D}_t)$",
            )
        self.h_list.append(h[0])
        return h

    def plot_exe_path_samples(self, exe_path_list):
        """Plot execution path samples."""

        # Optionally crop, given self.n_path_max
        exe_path_list = self.reduce_samp_list(exe_path_list)

        # Reset color cycle
        cgen = itertools.cycle(self.clist)

        for exe_path in exe_path_list:
            h = self.ax.plot(
                exe_path.x,
                exe_path.y,
                ".",
                #color=next(cgen)['color'],
                color="#d62728",
                markersize=3,
                linewidth=0.5,
                label="$\{ \\tilde{e}_\mathcal{A}^j \} \sim p(e_\mathcal{A}(f) | \mathcal{D}_t)$",
            )
        self.h_list.append(h[0])
        return h

    def plot_model_data(self, data):
        """Plot data, assumed to have attributes x and y."""
        label = "$\mathcal{D}_t = \{(x_i, y_i)\}_{i=1}^t$"
        #label = "Observations"
        #h = self.ax.plot(data.x, data.y, "o", color="deeppink", label=label)
        h = self.ax.plot(data.x, data.y, "o", color="black", label=label)
        self.h_list.append(h[0])
        return h

    def plot_acqfunction(self, x_test, acq_list):
        """Plot acquisition function as new subplot Axes."""
        self.acqfunction_shown = True

        # Update fig size
        fig_height = self.fig.get_figheight()
        add_height = 1.0
        self.fig.set_figheight(fig_height + add_height)

        # Shift current axes up, re-adjust height
        l, b, w, h = self.ax.get_position().bounds
        # -----
        #print('ax current bounds:')
        #print([l, b, w, h])
        # -----
        h_ratio = fig_height / (fig_height + add_height) 
        new_h = h * h_ratio
        new_b = b + h - new_h
        self.ax.set_position([l, new_b, w, new_h])

        # Add subplot for acqfunction plot
        self.ax_acq = self.fig.add_subplot(2, 1, (1, 2))

        # Adjust self.ax_acq
        l, b, w, h = self.ax_acq.get_position().bounds
        # -----
        #print('ax_acq current bounds:')
        #print([l, b, w, h])
        # -----
        gap_in = 0.15 # for 0.15 inch gap
        gap = h * gap_in / (fig_height + add_height)

        new_h_no_gap = h * add_height / (fig_height + add_height)
        new_h = new_h_no_gap - gap

        self.ax_acq.set_position([l, b, w, new_h])

        acq_arr = np.array(acq_list)
        h = self.ax_acq.plot(
            np.array(x_test).reshape(-1),
            acq_arr,
            "--",
            color="#ff7f0e",
            #color="red",
            linewidth=1,
            label="$\\alpha_t(x)$",
        )

        self.h_list.append(h[0])
        return h

    def plot_acqoptima(self, acq_list, x_test):
        """Plot optima of acquisition function."""
        acq_opt = x_test[np.argmax(acq_list)]

        # Set ylims for vertical lines
        if self.params.lims:
            ylim_ax = self.params.lims[2:]
        else:
            ylim_ax = self.ax.get_ylim()
        ylim_ax_acq = self.ax_acq.get_ylim()

        h = self.ax.plot(
            [acq_opt, acq_opt],
            ylim_ax,
            '--',
            color="black",
            label="$x_t = $ argmax$_{x \in \mathcal{X}}$ $\\alpha_t(x)$",
        )
        self.ax_acq.plot([acq_opt, acq_opt], ylim_ax_acq, '--', color="black")

        self.h_list.append(h[0])
        return h

    def plot_postpred_given_exe_path_samples(self, x_test, mu_list, std_list):
        """Plot posterior predictive given execution path sample, for each sample."""

        # Optionally crop, given self.n_path_max
        mu_list = self.reduce_samp_list(mu_list)
        std_list = self.reduce_samp_list(std_list)

        for mu_samp, std_samp in zip(mu_list, std_list):
            lcb = mu_samp - 3 * std_samp
            ucb = mu_samp + 3 * std_samp
            h = self.ax.fill_between(
                np.array(x_test).reshape(-1),
                lcb,
                ucb,
                color="blue",
                alpha=0.05,
                label="$p(y|\mathcal{D}_t, \\tilde{e}_\mathcal{A}^j, x)$",
            )
        self.h_list.append(h)
        return h

    def plot_post_f_samples(self, model, x_test, exe_path_list):
        """Compute and then plot posterior function samples."""

        # Optionally crop, given self.n_path_max
        exe_path_list = self.reduce_samp_list(exe_path_list)

        # Reset color cycle
        cgen = itertools.cycle(self.clist)

        for exe_path in exe_path_list:
            comb_data = Namespace()
            comb_data.x = model.data.x + exe_path.x
            comb_data.y = model.data.y + exe_path.y
            mu, cov = model.gp_post_wrapper(x_test, comb_data, full_cov=True)
            f_sample = model.get_normal_samples(mu, cov, 1, full_cov=True)
            f_sample = np.array(f_sample).reshape(-1)

            h = self.ax.plot(
                np.array(x_test).reshape(-1),
                f_sample,
                "-",
                #color=next(cgen)['color'],
                color="#d62728",
                alpha=0.5,
                linewidth=0.5,
                label="$\{\\tilde{f}\} \sim p(f | \mathcal{D}_t)$",
            )

        #self.h_list.append(h[0])
        return h

    def plot_post_f_samples_exe_path_postpred_means(self, x_test, mu_list):
        """
        Plot execution path posterior predictive means as approximate posterior function
        samples.
        """

        # Optionally crop, given self.n_path_max
        mu_list = self.reduce_samp_list(mu_list)

        # TODO: consider whether following are true posterior samples.
        for mu_samp in mu_list:
            h = self.ax.plot(
                np.array(x_test).reshape(-1),
                mu_samp,
                "-",
                alpha=0.5,
                linewidth=0.5,
                label="$\{\\tilde{f}\} \sim p(f | \mathcal{D}_t)$",
            )
        #self.h_list.append(h[0])
        return h

    def plot_acq_out_cluster(
        self,
        model,
        exe_path_list,
        output_list,
        acq_list,
        x_test,
        mu,
        std,
        mu_list,
        std_list,
        cluster_idx_list,
        mean_cluster_list,
        std_cluster_list,
    ):
        """
        Visualize the acquisition function, optimization, and related details, for a 1D
        continuous domain.
        """
        # Plot various details
        h1 = self.plot_postpred(x_test, mu, std)
        h2 = self.plot_clusters(
            x_test,
            cluster_idx_list,
            mean_cluster_list,
            std_cluster_list,
            output_list,
            exe_path_list,
        )
        h3 = self.plot_acqfunction(x_test, acq_list)
        h4 = self.plot_model_data(model.data)
        h5 = self.plot_acqoptima(acq_list, x_test)

        ## Legend
        h_list = [h4[0], h1, h2[0], h5[0], h3[0]]
        self.make_legend(h_list)

    def plot_clusters(
        self,
        x_test,
        cluster_idx_list,
        mean_cluster_list,
        std_cluster_list,
        output_list,
        exe_path_list,
    ):
        """Plot clusters of execution paths."""

        # Reset color cycle
        cgen = itertools.cycle(self.clist)

        # Loop through clusters
        mean_std_idx_list = zip(mean_cluster_list, std_cluster_list, cluster_idx_list)
        for mean_cluster, std_cluster, cluster_idx in mean_std_idx_list:
            nextcolor = next(cgen)['color']

            # Plot execution paths in each cluster
            self.plot_cluster_exe_paths(cluster_idx, exe_path_list, nextcolor)

            # Plot means of each cluster
            h = self.plot_cluster_means(x_test, mean_cluster, cluster_idx, nextcolor)

            # Plot stds of each cluster
            #self.plot_cluster_stds(x_test, mean_cluster, std_cluster, nextcolor)

            # Plot cluster properties
            self.plot_cluster_property(cluster_idx, output_list, nextcolor)

        return h

    def plot_cluster_exe_paths(self, cluster_idx, exe_path_list, color):
        """Plot execution paths for samples in each cluster."""
        exe_path_subset = [exe_path_list[i] for i in cluster_idx]
        x_offset = np.random.uniform(-0.1, 0.1)
        for exe_path in exe_path_subset:
            h = plt.plot(
                np.array(exe_path.x) + x_offset,
                exe_path.y,
                ".",
                color=color,
                markersize=4,
                linewidth=0.5,
            )
        return h

    def plot_cluster_means(self, x_test, mean_cluster, cluster_idx, color):
        """Plot mean of sampled functions for each cluster."""
        h = plt.plot(
            np.array(x_test).reshape(-1),
            mean_cluster,
            '--',
            #linewidth=0.05 * len(cluster_idx),
            linewidth=np.log(len(cluster_idx)) + 0.5,
            color=color,
            label="Cluster posterior predictive",
        )
        return h

    def plot_cluster_stds(self, x_test, mean_cluster, std_cluster, color):
        """Plot std of sampled functions for each cluster."""
        lcb = mean_cluster - 3 * std_cluster
        ucb = mean_cluster + 3 * std_cluster
        h = plt.fill_between(
            np.array(x_test).reshape(-1),
            lcb,
            ucb,
            alpha=0.1,
            color=color,
            label="Cluster posterior predictive",
        )
        return h

    def plot_cluster_property(self, cluster_idx, output_list, color):
        """Plot property of interest for samples in each cluster."""
        output_subset = [output_list[i] for i in cluster_idx]
        for output in output_subset:
            y_offset = np.random.uniform(-0.2, 0.2)
            h = plt.plot(
                output,
                5.0 + y_offset,
                'o',
                color=color,
                label="argmax output"
            )
        return h

    def make_legend(self):
        """Make the legend."""

        fig_height = self.fig.get_figheight()
        l, b, w, h = self.ax.get_position().bounds
        #print('legend: ax current bounds:')
        #print([l, b, w, h])

        #gap_in = 1.0
        #gap = h  * gap_in / fig_height
        #gap = gap_in * fig_height
        #gap = gap_in * h / fig_height
        #gap = 0.125 + b - 0.27
        #print(f'GAP IS: {gap}')

        n_leg = len(self.h_list)
        n_col = 3
        n_row = 1 if n_leg <= n_col else 2
        #bot = 0.15 if n_row == 1 else 0.21
        #bot = gap if n_row == 1 else gap

        # For legend above axes
        #bot = 0.0
        #bot = 0.1 * h
        #bot = 0.1 / (h - b)
        #bbta = (0, bot, 1, 1)
        bbta = (0, 0.25, 1, 1)
        #h_new = 0.1 * h + 0.05
        #bbta = (0, 1.0, 1, h_new)

        loc = "upper center"

        # Draw legend
        leg = self.ax.legend(
            handles=self.h_list,
            loc=loc,
            bbox_to_anchor=bbta,
            ncol=n_col,
            mode="expand",
        )
        return leg

    def set_post_plot_details(self):
        """Set post plot details."""
        # x axis
        if getattr(self, "acqfunction_shown", False):
            self.ax_acq.set_xlabel(self.params.xlabel)
            #self.ax.get_xaxis().set_ticklabels([])
            self.ax.get_xaxis().set_visible(False)
        else:
            self.ax.set_xlabel(self.params.xlabel)

        # y axis
        if getattr(self, "acqfunction_shown", False):
            #self.ax_acq.get_yaxis().set_ticklabels([])
            self.ax_acq.get_yaxis().set_visible(False)
        self.ax.set_ylabel(self.params.ylabel)

        # lims
        if self.params.lims:
            self.ax.set_xlim(self.params.lims[0], self.params.lims[1])
            self.ax_acq.set_xlim(self.params.lims[0], self.params.lims[1])
            self.ax.set_ylim(self.params.lims[2], self.params.lims[3])

    def reduce_samp_list(self, samp_list):
        """Optionally reduce list of samples, based on self.n_path_max."""
        if self.params.n_path_max:
            samp_list = samp_list[:self.params.n_path_max]

        return samp_list
