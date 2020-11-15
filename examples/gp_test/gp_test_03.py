import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt

from bax.models.simple_gp import SimpleGp
from bax.models.function import FunctionSample
from bax.alg.algorithms import LinearScan
from bax.util.timing import Timer

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
data.x = [[1.0], [2.0], [3.0], [10.0]]
data.y = [f(x[0]) for x in data.x]
model.set_data(data)

# Set function sample with model
fs = FunctionSample(verbose=False)
fs.set_model(model)

# Set algorithm
algo = LinearScan()

def sample_exe_path():
    """Return execution path sample."""
    fs.reset_query_history()
    exe_path, _ = algo.run_algorithm_on_f(fs)
    return exe_path

# Sample and plot execution paths
n_paths = 12

fig = plt.figure(figsize=(10, 5))

with Timer(f'Sample {n_paths} execution paths'):
    for _ in range(n_paths):
        exe_path_sample = sample_exe_path()

        # Plot execution path sample
        plt.plot(
            exe_path_sample.x, exe_path_sample.y, '.-', markersize=4, linewidth=0.5
        )

# Plot data and true execution path
plt.plot(exe_path_true.x, exe_path_true.y, '-', color='k', linewidth=3)
plt.plot(data.x, data.y, 'o', color='deeppink')

plt.xlim([0, 21])
plt.ylim([-4, 4])

plt.xlabel('x')
plt.ylabel('y')

plt.gca().set_aspect('equal', adjustable='box')

#neatplot.save_figure('testo', 'pdf')

plt.show()
