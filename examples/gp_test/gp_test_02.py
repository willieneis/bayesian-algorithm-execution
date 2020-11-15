import copy
import numpy as np
import matplotlib.pyplot as plt

from bax.models.simple_gp import SimpleGp
from bax.models.function import FunctionSample
from bax.alg.algorithms import LinearScan
from bax.util.timing import Timer

#import neatplot

seed = 15
np.random.seed(seed)

# Instantiate GP with data
gp = SimpleGp({'ls': 2.0, 'alpha': 1.5})
data = {'x': [[1.0], [2.0], [3.0]], 'y': [-0.5, 0.0, 0.5]}
#data = {'x': [[1], [2], [3], [12]], 'y': [-0.5, 0, 0.5, 0]} # Add pt at x=12
gp.set_data(data)

# Set function sample with model
fs = FunctionSample(verbose=False)
fs.set_model(gp)

# Set algorithm
algo = LinearScan()
algo.set_function(fs)

def sample_exe_path():
    """Return execution path sample."""
    fs.reset_query_history()
    exe_path, _ = algo.run_algorithm()
    return exe_path

# Define "true execution path"
exe_path_true = sample_exe_path()

# Sample and plot execution paths
n_paths = 5

fig = plt.figure(figsize=(10, 5))

with Timer(f'Sample {n_paths} paths'):
    for _ in range(n_paths):
        exe_path_sample = sample_exe_path()

        # Plot execution path sample
        n_data = len(data['x'])
        plt.plot(exe_path_sample.x[n_data-1:], exe_path_sample.y[n_data-1:], 'o--')

# Plot data and true execution path
plt.plot(data['x'], data['y'], 'o--', color='k')
plt.plot(exe_path_true.x, exe_path_true.y, 'o--', color='k')

plt.xlim([0, 21])
plt.ylim([-4, 4])

plt.xlabel('x')
plt.ylabel('y')

plt.gca().set_aspect('equal', adjustable='box')

#neatplot.save_figure('testo', 'pdf')

plt.show()
