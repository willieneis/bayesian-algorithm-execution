import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from bax.models.simple_gp import SimpleGp
from bax.alg.evolution_strategies import EvolutionStrategies
from bax.acq.acqoptimize import AcqOptimizer

from branin import branin, unif_random_sample_domain

import neatplot
neatplot.set_style('fonts')


seed = 11
np.random.seed(seed)

# Set function
f = lambda x: -1 * branin(x)
domain = [[-5, 10], [0, 15]]
#init_x = [7.0, 15.0]
init_x = [4.0, 14.0]

# Set data for model
data = Namespace()
data.x = [init_x]
data.y = [f(x) for x in data.x]

# Set model as a GP
gp_params = {'ls': 3.0, 'alpha': 2.0, 'sigma': 1e-2}
model = SimpleGp(gp_params)
model.set_data(data)

# Set algorithm
algo = EvolutionStrategies(
    {'n_generation': 15, 'n_population': 8, 'samp_str': 'mut', 'init_x': init_x}
) # Code currently requires init to 0

# Pre-computed algorithm output on f:
#algo_output_f = 10 # TODO: fix this

n_iter = 40

for i in range(n_iter):
    # Optimize acquisition function
    # TODO: sample x_test here
    x_test = unif_random_sample_domain(domain, n=150)
    acqopt = AcqOptimizer(
        {'x_test': x_test, 'acq_str': 'out', 'n_path': 50, 'viz_acq': False}
    )
    x_next = acqopt.optimize(model, algo)

    # Update algo.init_x
    algo.params.init_x = data.x[np.argmax(data.y)]

    # Compute current expected output
    expected_output = np.mean(acqopt.get_last_output_list(), 0)
    #out_abs_err = np.abs(algo_output_f - expected_output)

    # Query function, update data
    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)

    # Print
    print(f'Acq optimizer x_next = {x_next}')
    print(f'Current expected_output = {expected_output}')
    f_expected_output = f(expected_output)
    print(f'Current expected value f(expected_output) = {f_expected_output}')
    #print(f'Current output abs. error = {out_abs_err}')
    print(f'Finished iter i = {i}')

    #inp = input('Press enter to continue (any other key to stop): ')
    #if inp:
        #break

    # Update model
    model = SimpleGp(gp_params)
    model.set_data(data)
