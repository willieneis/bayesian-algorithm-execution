import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#plt.ion()
#import tensorflow as tf

from bax.alg.algorithms_new import TopK
from bax.models.simple_gp import SimpleGp
#from bax.models.gpfs_gp import GpfsGp
from bax.models.stan_gp import get_stangp_hypers
from bax.acq.acquisition_new import BaxAcqFunction
from bax.acq.acqoptimize_new import AcqOptimizer
from bax.acq.visualize_new import AcqViz1D

import neatplot
neatplot.set_style("fonts")
neatplot.update_rc("figure.dpi", 150)


seed = 12
np.random.seed(seed)
#tf.random.set_seed(seed)

# Set function
f = lambda x: 2 * np.sin(x[0])

# Set algorithm  details
min_x = 3.5
max_x = 20.0
len_path = 20
x_path = [[x] for x in np.random.uniform(min_x, max_x, len_path)]
algo = TopK({"x_path": x_path, "k": 2})

# Set data for model
data = Namespace()
data.x = [[1.0], [4.05], [7.27], [10.3], [13.2], [17.0]]
noise_scale = 0.1
data.y = [f(x) + noise_scale * np.random.normal() for x in data.x]

# Set model details
gp_params = {"ls": 2.0, "alpha": 2.0, "sigma": noise_scale ** 2}
#gp_params = get_stangp_hypers(f, n_samp=200) # NOTE: can use StanGp to fit hypers
#modelclass = GpfsGp
modelclass = SimpleGp # NOTE: can use SimpleGp model

# Set fast_legend to True for producing the legend, and to False for producing the main
# figure without a legend
fast_legend = True

# Set acquisition details
if fast_legend:
    acqfn_params1 = {"acq_str": "exe", "n_path": 20, "crop": False}     # EIG 1
    acqfn_params2 = {                                                   # EIG 2
        "acq_str": "out",
        "crop": False,
        "n_path": 20,
        "min_neighbors": 0,
        "max_neighbors": 20,
        "dist_thresh": 5.00,
    }
    acqfn_params3 = {"acq_str": "exe", "n_path": 20, "crop": True}      # EIG 3

else:
    acqfn_params1 = {"acq_str": "exe", "n_path": 100, "crop": False}    # EIG 1
    acqfn_params2 = {                                                   # EIG 2
        "acq_str": "out",
        "crop": False,
        "n_path": 1100,
        "min_neighbors": 3,
        "max_neighbors": 20,
        "dist_thresh": 0.05,
    }
    acqfn_params3 = {"acq_str": "exe", "n_path": 100, "crop": True}     # EIG 3

acqfn_params_list = [acqfn_params1, acqfn_params2, acqfn_params3]

x_test = [[x] for x in np.linspace(0.0, max_x, 500)]
y_test = [f(x) for x in x_test]
acqopt_params = {"x_batch": x_test}






#### ------------------------
####        BAX BELOW
#### ------------------------

# Set model
model = modelclass(gp_params, data)

# Set and optimize acquisition function
acqfn_list = []
x_next_list = []
for acqfn_params in acqfn_params_list:
    acqfn_list.append(BaxAcqFunction(acqfn_params, model, algo))
    acqopt = AcqOptimizer(acqopt_params)
    x_next_list.append(acqopt.optimize(acqfn_list[-1]))


# Normalize each acq_list to have minimum at 0
al_list_orig = [acqfn.acq_vars["acq_list"] for acqfn in acqfn_list]
al_list = []
for al in al_list_orig:
    al = np.array(al).reshape(-1)
    al = (al - np.min(al))
    al_list.append(al)

# Set acqfn to be EIG 3 (idx 2), and acqfn_full to be EIG 1 (idx 0)
acqfn = acqfn_list[2]
acqfn_full = acqfn_list[0]

# Compute current expected output
xl_list = []
for output in acqfn.output_list:
    xl = []
    list(map(xl.extend, output.x))
    xl_list.append(xl)
expected_output = np.mean(xl_list, 0)

# Plot
exe_path_list = acqfn.exe_path_list
exe_path_full_list = acqfn.exe_path_full_list
output_list = acqfn.output_list
mu = acqfn.acq_vars["mu"]
std = acqfn.acq_vars["std"]
mu_list = acqfn.acq_vars["mu_list"]
std_list = acqfn.acq_vars["std_list"]

#lims = (0, max_x, -5.5, 6.5)
#lims = (0, max_x, -5.5, 8.5)
lims = (0, max_x, -7.5, 6.0)

#vizzer_params = {"lims": lims, "n_path_max": 10, "figsize": (7, 2)}
vizzer_params = {"lims": lims, "n_path_max": 6, "figsize": (7, 2)}
vizzer = AcqViz1D(vizzer_params)

h_postpred = vizzer.plot_postpred(x_test, mu, std, noise=noise_scale)
h_acq_1 = vizzer.plot_acqfunction(x_test, al_list[1])
h_fsamp = vizzer.plot_post_f_samples(model, x_test, exe_path_full_list)
h_exepath = vizzer.plot_exe_path_samples(exe_path_full_list)
h_output = vizzer.plot_exe_path_crop_samples(exe_path_list)
#vizzer.plot_postpred_given_exe_path_samples(x_test, mu_list, std_list)
h_data = vizzer.plot_model_data(model.data)
vizzer.set_post_plot_details()
(fig, ax, ax_acq) = (vizzer.fig, vizzer.ax, vizzer.ax_acq)

# Plot exe path grey vertical bars
ylims_acq = ax_acq.get_ylim()
for x in x_path:
    h_greybar = ax.plot(
        #[x[0], x[0]], [lims[2], lims[3]], '-', color=(0, 0, 0, 0.075), linewidth=5.0, zorder=0,
        [x[0], x[0]], [lims[2], -5], '-', color=(0, 0, 0, 0.075), linewidth=5.0, zorder=0,
    )
    ax.plot(
        [x[0], x[0]], [lims[2], -5], '-', color=(0, 0, 0, 0.02), linewidth=5.0, zorder=10,
    )


# EIG-specific colors
# yelloworange = "#ff7f0e"
# red = "#d62728"
color_list = ["#2ca02c", "#ff7f0e", "#1f77b4"]

# Plot additional acq functions
h_acq_0 = ax_acq.plot(
    np.array(x_test).reshape(-1),
    al_list[0] * 0.8,   # Normalize scale slightly
    "--",
    color=color_list[0],
    linewidth=1,
    label="$\\alpha_t(x)$",
)

h_acq_2 = ax_acq.plot(
    np.array(x_test).reshape(-1),
    al_list[2],
    "--",
    color=color_list[2],
    linewidth=1,
    label="$\\alpha_t(x)$",
)

# Afterwards, plot acq optima
ylim_ax = ax.get_ylim()
ylim_ax_acq = ax_acq.get_ylim()
for idx in range(len(al_list)):
    al = al_list[idx]
    acq_opt = x_test[np.argmax(al)]
    h = ax.plot(
        [acq_opt, acq_opt],
        ylim_ax,
        '-',
        #color="black",
        color=color_list[idx],
        label="$x_t = $ argmax$_{x \in \mathcal{X}}$ $\\alpha_t(x)$",
    )
    ax_acq.plot([acq_opt, acq_opt], ylim_ax_acq, '-', color=color_list[idx])

ax_acq.set_ylim(ylim_ax_acq)

# Plot the true f
h_truef = ax.plot(x_test, y_test, "-", color="k", linewidth=2) # True f



# Legend
h_list = [
    h_truef[0],
    h_greybar[0],
    h_data[0],
    h_postpred,
    h_fsamp[0],
    h_exepath[0],
    h_output[0],
    h_acq_0[0],
    h_acq_1[0],
    h_acq_2[0],
]

label_list = [
    '$f$',
    #'$(z_1, \ldots, z_{|X|})$',
    '$x \in X$',
    '$(x, y_x) \in \mathcal{D}_t$',
    '$p(y_x|\mathcal{D}_t)$',
    '$\\widetilde{f} \sim p(f | \mathcal{D}_t)$',
    '$\\widetilde{e}_\mathcal{A} \sim p(e_\mathcal{A} | \mathcal{D}_t)$',
    '$\\widetilde{O}_\mathcal{A} \sim p(O_\mathcal{A} | \mathcal{D}_t)$',
    'EIG$^e_t(x)$',
    'EIG$_t(x)$',
    'EIG$^v_t(x)$',
]

if fast_legend:
    leg = ax.legend(
        h_list,
        label_list,
        #loc='center',
        loc='lower left',
        ncol=3,
        #ncol=2,
        #mode='expand',
        bbox_to_anchor=(1.0, 1.0),
    )
    leg.set_zorder(100)
    frame = leg.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor((0.7, 0.7, 0.7, 1))
    frame.set_alpha(1)





neatplot.save_figure(f"topk_viz", "pdf")
plt.show()




 # OLD PLOTTING CALL

#(ax, ax_acq) = vizzer.plot_acqoptimizer_all(
    #model,
    #acqfn.exe_path_list,
    #acqfn.output_list,
    #acqfn.acq_vars["acq_list"],
    #x_test,
    #acqfn.acq_vars["mu"],
    #acqfn.acq_vars["std"],
    #acqfn.acq_vars["mu_list"],
    #acqfn.acq_vars["std_list"],
#)
