import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
#plt.ion()

from bax.alg.evolution_strategies import EvolutionStrategies
from bax.acq.visualize2d import AcqViz2D

from branin import branin, branin_xy

import neatplot
neatplot.set_style('fonts')
neatplot.update_rc('font.size', 20)


seed = 11
np.random.seed(seed)


# Set function
f = branin

# Set algorithm details
init_x = [4.8, 13.0]
#init_x = [6.0, 10.0] # Center-right start

domain = [[-5, 10], [0, 15]]

algo_params = {
    'n_generation': 26,
    'n_population': 8,
    'samp_str': 'mut',
    'opt_mode': 'min',
    'init_x': init_x,
    'domain': domain,
    'normal_scale': 0.5,
    'keep_frac': 0.3,
    'crop': False,
}
algo = EvolutionStrategies(algo_params)

# Run algorithm on f
exe_path, output = algo.run_algorithm_on_f(f)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
vizzer = AcqViz2D(fig_ax=(fig, ax))
vizzer.plot_function_contour(branin_xy, domain)
#h1 = vizzer.plot_output_samples(acqfn.output_list)
h2 = vizzer.plot_model_data(exe_path)
h3 = vizzer.plot_expected_output(output)
h4 = vizzer.plot_optima([(-3.14, 12.275), (3.14, 2.275), (9.425, 2.475)])

# Legend
#vizzer.make_legend([h2[0], h3[0], h4[0]])

# Axis lims and labels
offset = 0.3
ax.set_xlim((domain[0][0] - offset, domain[0][1] + offset))
ax.set_ylim((domain[1][0] - offset, domain[1][1] + offset))
ax.set_aspect("equal", "box")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Evolution Strategy (Full)")

# Print
min_idx = np.argmin(exe_path.y)
print(f'Best point so far x*: {exe_path.x[min_idx]}')
print(f'Value of best point so far f(x*): {exe_path.y[min_idx]}')
print(f'Found at iter: {min_idx}')

# Save plot
neatplot.save_figure(f"branin_evo", "pdf")

# Show, pause, and close plot
plt.show()
