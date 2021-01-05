import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
#plt.ion()

from bax.models.simple_gp import SimpleGp
from bax.models.stan_gp import get_stangp_hypers
from bax.alg.evolution_strategies import EvolutionStrategies
from bax.acq.acqoptimize import AcqOptimizer
from bax.util.domain_util import unif_random_sample_domain

from branin import branin

import neatplot
neatplot.set_style('fonts')


seed = 11
np.random.seed(seed)

# Set function
f = branin
domain = [[-5, 10], [0, 15]]
init_x = [4.8, 13.0]
#init_x = [4.0, 14.0]
#init_x = [5.7, 13.25]
#init_x = [7.5, 13.0]

# Set data for model
data = Namespace()
data.x = [init_x]
data.y = [f(x) for x in data.x]

# Set model as a GP
gp_params = get_stangp_hypers(f, domain=domain, n_samp=500)
model = SimpleGp(gp_params)
model.set_data(data)

# Set algorithm
algo = EvolutionStrategies(
    {
        'n_generation': 15,
        'n_population': 8,
        'samp_str': 'mut',
        'init_x': init_x,
        'domain': domain,
        'normal_scale': 0.5,
        'keep_frac': 0.5,
    }
)

n_iter = 25

for i in range(n_iter):

    # Plot setup
    fig = plt.figure(figsize=(6, 6))

    plt.xlim([-5.1, 10.1])
    plt.ylim([-0.1, 15.1])

    plt.xlabel('x')
    plt.ylabel('y')

    im = plt.imread('examples/es/branin_contour.png')
    implot = plt.imshow(
        im, extent=[domain[0][0], domain[0][1], domain[1][0], domain[1][1]]
    )

    # Update algo.init_x
    algo.params.init_x = data.x[np.argmin(data.y)]

    # Optimize acquisition function
    x_test = unif_random_sample_domain(domain, n=150)
    acqopt = AcqOptimizer(
        {
            'x_test': x_test,
            'acq_str': 'out',
            'n_path': 20,
            'viz_acq': True,
            'viz_dim': 2,
        }
    )
    x_next = acqopt.optimize(model, algo)

    # Compute current expected output
    expected_output = np.mean(acqopt.get_last_output_list(), 0)

    # Query function, update data
    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)

    # Print
    print(f'Acq optimizer x_next = {x_next}')
    print(f'Current expected_output = {expected_output}')
    f_expected_output = f(expected_output)
    print(f'Current expected value f(expected_output) = {f_expected_output}')
    print(f'Finished iter i = {i}')

    # Show plot
    plt.show()

    inp = input('Press enter to continue (any other key to stop): ')
    if inp:
        break

    # Update model
    model = SimpleGp(gp_params)
    model.set_data(data)
