import copy
import numpy as np
import matplotlib.pyplot as plt

from bax.models.simple_gp import SimpleGp
from bax.models.exe_path import ExePath

#import neatplot

seed = 15
np.random.seed(seed)

# Instantiate GP with data
gp = SimpleGp({'ls': 2.0, 'alpha': 1.5, 'sigma': 1e-5})
data = {'x': [[1.0], [2.0], [3.0]], 'y': [-0.5, 0.0, 0.5]}
gp.set_data(data)

# Predefine execution x-path
x_path = [[x] for x in np.linspace(3.5, 20, 100)]

def sample_exe_path():
    """Return execution path sample."""
    ep = ExePath(verbose=False)
    ep.init_path_with_model(gp)
    for x in x_path:
        y = ep.next_step(x)

    return ep.exe_path


# Define "true execution path"
exe_path_true = sample_exe_path()

# Sample and plot execution paths
n_paths = 5

fig = plt.figure(figsize=(10, 5))

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
