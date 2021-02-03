import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
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


seed = 11
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
noise = 0.5
data.y = [f(x) + noise * np.random.random() for x in data.x]

# Set model details
gp_params = {"ls": 2.0, "alpha": 2.0, "sigma": 1e-2}
#gp_params = get_stangp_hypers(f, n_samp=200) # NOTE: can use StanGp to fit hypers
#modelclass = GpfsGp
modelclass = SimpleGp # NOTE: can use SimpleGp model

# Set acquisition details
acqfn_params1 = {"acq_str": "exe", "n_path": 100, "crop": False}    # EIG 1
acqfn_params2 = {                                                   # EIG 2
    "acq_str": "out",
    "crop": False,
    "n_path": 500,
    "min_neighbors": 5,
    "max_neighbors": 20,
    "dist_thresh": 0.05,
}
#acqfn_params2 = {                                                   # EIG 2
    #"acq_str": "out",
    #"crop": False,
    #"n_path": 20,
    #"min_neighbors": 0,
    #"max_neighbors": 20,
    #"dist_thresh": 5.00,
#}
acqfn_params3 = {"acq_str": "exe", "n_path": 100, "crop": True}     # EIG 3
acqfn_params_list = [acqfn_params1, acqfn_params2, acqfn_params3]
#acqfn_params_list = [acqfn_params2]

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

# Set acqfn to be idx 2 in list (EIG 3)
acqfn = acqfn_list[2]

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

lims = (0, max_x, -5.5, 6.5)

vizzer_params = {"lims": lims, "n_path_max": 5, "figsize": (7, 2)}
vizzer = AcqViz1D(vizzer_params)

vizzer.plot_postpred(x_test, mu, std, noise=0.1)
h_acq_1 = vizzer.plot_acqfunction(x_test, al_list[1])
vizzer.plot_exe_path_samples(exe_path_full_list)
vizzer.plot_exe_path_crop_samples(exe_path_list)
vizzer.plot_postpred_given_exe_path_samples(x_test, mu_list, std_list)
vizzer.plot_model_data(model.data)
vizzer.plot_post_f_samples(model, x_test, exe_path_full_list)
vizzer.set_post_plot_details()
(fig, ax, ax_acq) = (vizzer.fig, vizzer.ax, vizzer.ax_acq)

# Plot exe path grey vertical bars
ylims_acq = ax_acq.get_ylim()
for x in x_path:
    ax.plot(
        [x[0], x[0]], [lims[2], lims[3]], '-', color=(0, 0, 0, 0.1), linewidth=5.0, zorder=0,
    )
    ax.plot(
        [x[0], x[0]], [lims[2], lims[3]], '-', color=(0, 0, 0, 0.05), linewidth=5.0, zorder=10,
    )

# Plot additional acq functions
h_acq_0 = ax_acq.plot(
    np.array(x_test).reshape(-1),
    al_list[0] * 0.8,   # Normalize scale slightly
    "--",
    color="green",
    linewidth=1,
    label="$\\alpha_t(x)$",
)

h_acq_2 = ax_acq.plot(
    np.array(x_test).reshape(-1),
    al_list[2],
    "--",
    color="blue",
    linewidth=1,
    label="$\\alpha_t(x)$",
)

# Afterwards, plot acq optima
vizzer.plot_acqoptima(al_list[0], x_test)
vizzer.plot_acqoptima(al_list[1], x_test)
vizzer.plot_acqoptima(al_list[2], x_test)

# Legend for acq optima
#ax_acq.legend(
    #[h_acq_0[0], h_acq_1[0], h_acq_2[0]],
    #['EIG 1', 'EIG 2', 'EIG 3'],
    #loc='lower left',
#)

# Plot the true f
ax.plot(x_test, y_test, "-", color="k", linewidth=2) # True f

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
