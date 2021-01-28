"""
Code for visualizing acquisition functions and optimization.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


class AcqViz1D:
    """
    Class to visualize acquisition function optimization for one-dimensional x.
    """

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
        h0 = self.plot_exe_path_samples(exe_path_list)
        h1 = self.plot_postpred(x_test, mu, std)
        # h1b = self.plot_post_f_samples(x_test, mu_list)
        h2 = self.plot_postpred_given_exe_path_samples(x_test, mu_list, std_list)
        h3 = self.plot_acqfunction(x_test, acq_list)
        h4 = self.plot_model_data(model.data)
        h5 = self.plot_acqoptima(acq_list, x_test)

        ## Legend
        h_list = [h0[0], h4[0], h1, h2, h5[0], h3[0]]
        self.make_legend(h_list)

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

    def plot_exe_path_samples(self, exe_path_list):
        """Plot execution path samples."""
        for exe_path in exe_path_list:
            h = plt.plot(
                exe_path.x,
                exe_path.y,
                ".",
                markersize=4,
                linewidth=0.5,
                label="$\{ \\tilde{e}_\mathcal{A}^j \} \sim p(e_\mathcal{A}(f) | \mathcal{D}_t)$",
            )
        return h

    def plot_postpred(self, x_test, mu, std):
        """Plot posterior predictive distribution."""
        lcb = mu - 3 * std
        ucb = mu + 3 * std
        h = plt.fill_between(
            np.array(x_test).reshape(-1),
            lcb,
            ucb,
            color="orange",
            alpha=0.2,
            label="$p(y|\mathcal{D}_t, x)$",
        )
        return h

    def plot_postpred_given_exe_path_samples(self, x_test, mu_list, std_list):
        """Plot posterior predictive given execution path sample, for each sample."""
        for mu_samp, std_samp in zip(mu_list, std_list):
            lcb = mu_samp - 3 * std_samp
            ucb = mu_samp + 3 * std_samp
            h = plt.fill_between(
                np.array(x_test).reshape(-1),
                lcb,
                ucb,
                color="blue",
                alpha=0.1,
                label="$p(y|\mathcal{D}_t, \\tilde{e}_\mathcal{A}^j, x)$",
            )
        return h

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
        clist = rcParams['axes.prop_cycle']
        cgen = itertools.cycle(clist)

        # Loop through clusters
        mean_std_idx_list = zip(mean_cluster_list, std_cluster_list, cluster_idx_list)
        for mean_cluster, std_cluster, cluster_idx in mean_std_idx_list:
            nextcolor = next(cgen)['color']

            # Plot execution paths in each cluster
            h = self.plot_cluster_exe_paths(cluster_idx, exe_path_list, nextcolor)

            # Plot means of each cluster
            #h = self.plot_cluster_means(x_test, mean_cluster, cluster_idx, nextcolor)

            # Plot stds of each cluster
            #self.plot_cluster_stds(x_test, mean_cluster, std_cluster, nextcolor)

            # Plot cluster properties
            self.plot_cluster_property(cluster_idx, output_list, nextcolor)

        return h

    def plot_cluster_exe_paths(self, cluster_idx, exe_path_list, color):
        """Plot execution paths for samples in each cluster."""
        #exe_path_subset = [exe_path_list[i] for i in cluster_idx]
        exe_path_subset = [exe_path_list[cluster_idx[0]]]
        x_offset = np.random.uniform(-0.1, 0.1)
        for exe_path in exe_path_subset:
            h = plt.plot(
                np.array(exe_path.x) + x_offset,
                exe_path.y,
                ".-",
                color=color,
                markersize=4,
                linewidth=0.5,
                label="Cluster exe path"
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
        #output_subset = [output_list[i] for i in cluster_idx]
        output_subset = [output_list[cluster_idx[0]]]
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

    def plot_post_f_samples(self, x_test, mu_list):
        """Plot posterior function samples."""
        # TODO: consider whether following are true posterior samples.
        for mu_samp in mu_list:
            h = plt.plot(
                np.array(x_test).reshape(-1),
                mu_samp,
                "-",
                alpha=0.75,
                linewidth=0.5,
                label="$\{\\tilde{f}\} \sim p(f | \mathcal{D}_t)$",
            )
        return h

    def plot_model_data(self, data):
        """Plot data, assumed to have attributes x and y."""
        h = plt.plot(data.x, data.y, "o", color="deeppink", label="Observations")
        # -----
        # plt.plot([0, 20], [0,0], '--', color='k', linewidth=0.5)
        # for x, y in zip(data.x, data.y):
        # plt.plot([x, x], [0, y], '-', color='b', linewidth=0.5)
        # h = plt.plot(data.x, data.y, 'o', color='b')
        # -----
        return h

    def plot_acqfunction(self, x_test, acq_list):
        """Plot acquisition function and dividing line, and resize plot axes."""
        ylim = plt.gca().get_ylim()
        acq_arr = np.array(acq_list)
        min_acq = np.min(acq_arr)
        max_acq = np.max(acq_arr)
        ylim_diff = ylim[1] - ylim[0]
        acq_height = 0.33 * ylim_diff
        ylim_new_min = ylim[0] - acq_height
        acq_arr = (acq_arr - min_acq) / (max_acq - min_acq) * acq_height + ylim_new_min
        h = plt.plot(
            np.array(x_test).reshape(-1),
            acq_arr,
            "-",
            color="red",
            linewidth=1,
            label="Acquisition function $\\alpha_t(x)$",
        )

        # Reset y axis
        plt.ylim([ylim_new_min, ylim[1]])

        # Plot dividing line
        xlim = plt.gca().get_xlim()
        plt.plot(xlim, [ylim[0], ylim[0]], "--", color="grey", alpha=0.8)

        return h

    def plot_acqoptima(self, acq_list, x_test):
        """Plot optima of acquisition function."""
        acq_opt = x_test[np.argmax(acq_list)]
        ylim = plt.gca().get_ylim()
        h = plt.plot(
            [acq_opt, acq_opt],
            ylim,
            '--',
            color="black",
            label="$x_t = $ argmax$_{x \in \mathcal{X}}$ $\\alpha_t(x)$",
        )
        return h

    def make_legend(self, h_list):
        """Make the legend."""

        # For legend within axes
        #bbta = None
        #loc = 1
        #ncol = 1

        # For legend above axes
        ax = plt.gca()
        ax.set_position([0.1, 0.1, 0.85, 0.7])
        bbta = (0.5, 1.24)
        loc = "upper center"
        ncol = 3

        # Draw legend
        lgd = plt.legend(handles=h_list, loc=loc, bbox_to_anchor=bbta, ncol=ncol)


class AcqViz2D:
    """
    Class to visualize acquisition function optimization for two-dimensional x.
    """

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
        h0 = self.plot_exe_path_samples(exe_path_list, output_list)
        h0b = self.plot_expected_output(output_list)
        #h1 = self.plot_postpred(x_test, mu, std)
        # h1b = self.plot_post_f_samples(x_test, mu_list)
        #h2 = self.plot_postpred_given_exe_path_samples(x_test, mu_list, std_list)
        #h3 = self.plot_acqfunction(x_test, acq_list)
        h4 = self.plot_model_data(model.data)
        h5 = self.plot_acqoptima(acq_list, x_test)

        ## Legend
        #h_list = [h0[0], h4[0], h1, h2, h5[0], h3[0]]
        h_list = [h0[0], h4[0], h0b[0], h5[0]]
        self.make_legend(h_list)

    def plot_exe_path_samples(self, exe_path_list, output_list):
        """
        Plot execution path and output samples. This method assumes an optimization
        algorithm that returns 2d locations.
        """

        # reset color cycle
        clist = rcParams['axes.prop_cycle']
        cgen = itertools.cycle(clist)

        # Plot exe_paths
        for exe_path in exe_path_list:
            nextcolor = next(cgen)['color']

            x_list = [xin[0] for xin in exe_path.x]
            y_list = [xin[1] for xin in exe_path.x]
            h = plt.plot(
                x_list,
                y_list,
                ".",
                color=nextcolor,
                markersize=3,
                linewidth=0.5,
                label="$\{ \\tilde{e}_\mathcal{A}^j \} \sim p(e_\mathcal{A}(f) | \mathcal{D}_t)$",
            )

            plt.plot(
                x_list,
                y_list,
                "-",
                color=nextcolor,
                markersize=3,
                linewidth=0.5,
                alpha=0.2,
            )

        # Plot exe_paths start, end, and best points
        plt.gca().set_prop_cycle(None)
        for exe_path, output in zip(exe_path_list, output_list):
            nextcolor = next(cgen)['color']
            plt.plot(exe_path.x[0][0], exe_path.x[0][1], "o", color='black', markersize=8)
            plt.plot(exe_path.x[0][0], exe_path.x[0][1], "o", color=nextcolor, markersize=7)
            plt.plot(output[0], output[1], "*", color='black', markersize=10)
            plt.plot(output[0], output[1], "*", color=nextcolor, markersize=6)

        return h

    def plot_output_samples(self, output_list):
        """
        Plot algorithm outputs. This method assumes an optimization algorithm that
        returns 2d locations.
        """

        # reset color cycle
        clist = rcParams["axes.prop_cycle"]
        cgen = itertools.cycle(clist)

        plt.gca().set_prop_cycle(None)
        for output in output_list:
            nextcolor = next(cgen)["color"]
            h = plt.plot(
                output[0],
                output[1],
                "x",
                color="blue",
                #color=nextcolor,
                markersize=4,
                label="$\{ \\tilde{o}_\mathcal{A}^j \} \sim  p(o_\mathcal{A} | \mathcal{D}_t)$",
            )

        return h

    def plot_expected_output(self, output_list):
        """Plot expected output."""
        expected_output = np.mean(output_list, 0)
        plt.plot(
            expected_output[0],
            expected_output[1],
            "*",
            color='black',
            markersize=11,
            #color='white',
            #markersize=12,
        )
        h = plt.plot(
            expected_output[0],
            expected_output[1],
            "*",
            color='green',
            markersize=9,
            #color='black',
            #markersize=8,
            label="$ \mathrm{E}[o_\mathcal{A}]$"
        )
        return h

    def plot_post_f_samples(self, x_test, mu_list):
        """Plot posterior function samples."""
        # TODO: consider whether following are true posterior samples.
        for mu_samp in mu_list:
            h = plt.plot(
                np.array(x_test).reshape(-1),
                mu_samp,
                "-",
                alpha=0.75,
                linewidth=0.5,
                label="$\{\\tilde{f}\} \sim p(f | \mathcal{D}_t)$",
            )
        return h

    def plot_model_data(self, data):
        """Plot data, assumed to have attributes x and y."""
        x_list = [xin[0] for xin in data.x]
        y_list = [xin[1] for xin in data.x]
        h = plt.plot(
            x_list, y_list, "o", color="black", markersize=8
        )
        #h = plt.plot(
            #x_list, y_list, "o", color="white", markersize=8
        #)
        h = plt.plot(
            x_list, y_list, "o", color="deeppink", label="$\mathcal{D}_t = \{x_i, y_i\}_{i=1}^t$", markersize=7
        )
        return h

    def plot_acqoptima(self, acq_list, x_test):
        """Plot optima of acquisition function."""
        acq_opt = x_test[np.argmax(acq_list)]
        h = plt.plot(
            acq_opt[0],
            acq_opt[1],
            "x",
            color="white",
            markersize=10,
            markeredgewidth=2,
            label="$x_t = $ argmax$_{x \in \mathcal{X}}$ $\\alpha_t(x)$",
        )
        h = plt.plot(
            acq_opt[0],
            acq_opt[1],
            "x",
            color="b",
            markersize=10,
            markeredgewidth=1.5,
            label="$x_t = $ argmax$_{x \in \mathcal{X}}$ $\\alpha_t(x)$",
        )
        return h

    def make_legend(self, h_list):
        """Make the legend."""

        # For legend within axes
        #bbta = None
        #loc = 1
        #ncol = 1

        # For legend above axes
        ax = plt.gca()
        ax.set_position([0.1, 0.1, 0.85, 0.7])
        bbta = (0.5, 1.24)
        loc = "upper center"
        ncol = 3

        # Draw legend
        lgd = plt.legend(handles=h_list, loc=loc, bbox_to_anchor=bbta, ncol=ncol)
