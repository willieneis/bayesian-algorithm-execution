import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt

from bax.models.simple_gp import SimpleGp
from bax.models.function import FunctionSample
from bax.alg.algorithms import LinearScan
from bax.util.timing import Timer
from bax.acq.acquisition import AcqFunction

#import neatplot

seed = 15
np.random.seed(seed)

# Set function
f = np.sin

# Set "true execution path"
exe_path_true = Namespace()
exe_path_true.x = [[x] for x in np.linspace(0.0, 20.0, 150)]
exe_path_true.y = [f(x[0]) for x in exe_path_true.x]

# Set model as a GP
model = SimpleGp({'ls': 2.0, 'alpha': 1.5})

# Set data for model
data = Namespace()
#data.x = [[1.0], [2.0], [3.0], [10.0]]
data.x = [[1.0], [2.0], [3.0], [10.0], [20], [14.7764], [6.9824]]
#data.x = [[1.0], [2.0], [3.0], [10.0], [20], [14.7764], [6.9824], [12.206], [18.259]]
data.y = [f(x[0]) for x in data.x]
model.set_data(data)

# Set function sample with model
fs = FunctionSample(verbose=False)
fs.set_model(model)

# Set algorithm
algo = LinearScan({'x_path': [[x] for x in np.linspace(3.5, 20, 20)]})

def sample_exe_path():
    """Return execution path sample."""
    fs.reset_query_history()
    exe_path, _ = algo.run_algorithm_on_f(fs)
    return exe_path

# Sample and plot execution paths
n_path = 50

fig = plt.figure(figsize=(10, 5))

with Timer(f'Sample {n_path} execution paths'):
    exe_path_list = []
    for _ in range(n_path):
        exe_path_sample = sample_exe_path()
        exe_path_list.append(exe_path_sample)

        # Plot execution path sample
        plt.plot(
            exe_path_sample.x, exe_path_sample.y, '.-', markersize=4, linewidth=0.5
        )

# Plot data and true execution path
plt.plot(exe_path_true.x, exe_path_true.y, '-', color='k', linewidth=3)
plt.plot(data.x, data.y, 'o', color='deeppink')

# Plot settings
plt.xlim([0, 21])
plt.ylim([-4, 4])

plt.xlabel('x')
plt.ylabel('y')

plt.gca().set_aspect('equal', adjustable='box')


n_test = 200
with Timer(f'Compute acquisition at {n_test} test points'):
    x_test = [[x] for x in np.linspace(3.5, 20, 200)]
    # Compute mean and std arrays for posterior
    mu, std = model.get_post_mu_cov(x_test, full_cov=False)

    # Compute mean and std arrays for posterior given execution path samples
    mu_list = []
    std_list = []
    for exe_path in exe_path_list:
        fs.set_query_history(exe_path)
        mu_samp, std_samp = fs.get_post_mean_std_list(x_test)
        mu_list.append(mu_samp)
        std_list.append(std_samp)
        # ---
        lcb = mu_samp - 3 * std_samp
        ucb = mu_samp + 3 * std_samp
        plt.fill_between(np.array(x_test).reshape(-1), lcb, ucb, color='blue', alpha=0.9)

    lcb = mu - 3 * std
    ucb = mu + 3 * std
    plt.fill_between(np.array(x_test).reshape(-1), lcb, ucb, color='blue', alpha=0.1)

# Compute acquisition function
acqf = AcqFunction({'acq_str': 'exe'})
acq_list = acqf(std, std_list)
acq_arr = np.array(acq_list)

acq_opt_idx = np.argmax(acq_arr)
acq_opt = x_test[acq_opt_idx]
print(f'Acq optimizer = {acq_opt}')

# Plot acquisition function
ylim = plt.gca().get_ylim()

min_acq = np.min(acq_arr)
max_acq = np.max(acq_arr)
ylim_diff = ylim[1] - ylim[0]
acq_height = 0.33 * ylim_diff
ylim_new_min = ylim[0] - acq_height
acq_arr = (acq_arr - min_acq) / (max_acq - min_acq) * acq_height + ylim_new_min
plt.plot(np.array(x_test).reshape(-1), acq_arr, '-', color='red', linewidth=1)

plt.ylim([ylim_new_min, ylim[1]])

# Plot dividing line
xlim = plt.gca().get_xlim()
plt.plot(xlim, [ylim[0], ylim[0]], '--', color='k')


# Save figure
#neatplot.save_figure('testo', 'pdf')

# Show plot
plt.show()
