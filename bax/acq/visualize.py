"""
Code for visualizing acquisition functions and optimization.
"""
import numpy as np
import matplotlib.pyplot as plt

class AcqViz1D:
    """
    Class to visualize acquisition function optimization for one-dimensional x.
    """

    def plot_acqoptimizer_all(
        self, model, exe_path_list, acq_list, x_test, mu, std, mu_list, std_list
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
        h4 = self.plot_model_data(model)
        h5 = self.plot_acqoptima(acq_list, x_test)

        ## Legend
        h_list = [h0[0], h4[0], h1, h2, h5[0], h3[0]]
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

    def plot_model_data(self, model):
        """Plot model.data."""
        h = plt.plot(
            model.data.x, model.data.y, "o", color="deeppink", label="Observations"
        )
        # -----
        # plt.plot([0, 20], [0,0], '--', color='k', linewidth=0.5)
        # for x, y in zip(model.data.x, model.data.y):
        # plt.plot([x, x], [0, y], '-', color='b', linewidth=0.5)
        # h = plt.plot(model.data.x, model.data.y, 'o', color='b')
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
