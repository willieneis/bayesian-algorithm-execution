import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from bax.models.simple_gp import SimpleGp
from bax.alg.algorithms import AverageOutputs
from bax.acq.acqoptimize import AcqOptimizer

import neatplot
neatplot.set_style('fonts')


seed = 11
np.random.seed(seed)

# Set function
f = lambda x: 2 * np.sin(x[0])

# Set data for model
data = Namespace()
data.x = [[1.0]]
data.y = [f(x) for x in data.x]

# Set model as a GP
gp_params = {'ls': 2.0, 'alpha': 2.0, 'sigma': 1e-2}
model = SimpleGp(gp_params)
model.set_data(data)

# Set arrays
min_x = 3.5
max_x = 20.0
len_path = 15
x_rand = np.random.uniform(min_x, max_x, len_path)
x_path = [[x] for x in x_rand]
x_path_big = [[x] for x in np.linspace(0.0, max_x, 200)]
x_test = [[x] for x in np.linspace(0.0, max_x, 500)]

# Set algorithm
algo = AverageOutputs({'x_path': x_path})

# Set "true execution path"
exe_path_true = Namespace(x=x_path_big, y=[f(x) for x in x_path_big])


n_iter = 40

for i in range(n_iter):
    # Plot setup
    fig = plt.figure(figsize=(8, 5))
    plt.xlim([0, max_x + 1])
    plt.ylim([-5.5, 6.5])
    plt.xlabel('x')
    plt.ylabel('y')

    # Optimize acquisition function
    acqopt = AcqOptimizer({'x_test': x_test})
    x_next = acqopt.optimize(model, algo)
    print(f'Acq optimizer x_next = {x_next}')
    print(f'Finished iter i = {i}')

    # Plot true execution path
    plt.plot(exe_path_true.x, exe_path_true.y, '-', color='k', linewidth=3)

    # Show plot
    #neatplot.save_figure(f'gp_test_10_{i}')
    plt.show()

    inp = input('Press enter to continue (any other key to stop): ')
    if inp:
        break
    plt.close()

    # Query function, update data
    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)

    # Update model
    model = SimpleGp(gp_params)
    model.set_data(data)
