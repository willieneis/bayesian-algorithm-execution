import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from bax.models.simple_gp import SimpleGp
from bax.alg.algorithms import AverageOutputs
from bax.acq.acqoptimize import AcqOptimizer


seed = 11
np.random.seed(seed)

# Set function
f = lambda x: 2 * np.sin(x[0])

# Set data for model
data = Namespace()
#data.x = [[1.0], [2.0], [3.0]]
data.x = [[1.0]]
data.y = [f(x) for x in data.x]

# Set model as a GP
gp_params = {'ls': 2.0, 'alpha': 2.0, 'sigma': 1e-2}
model = SimpleGp(gp_params)
model.set_data(data)

# Set algorithm
x_path = [
[5.1], [5.3], [5.5],
[7.1], [7.3], [7.5],
[11.1], [11.2], [11.9],
[15.1], [16.3], [18.5],
]
algo = AverageOutputs({'x_path': x_path})

# Set "true execution path"
x_path_big = [[x] for x in np.linspace(0, 20, 200)]
exe_path_true = Namespace(x=x_path_big, y=[f(x) for x in x_path_big])


n_iter = 40

for i in range(n_iter):
    # Plot setup
    fig = plt.figure(figsize=(8, 5))
    #plt.xlim([0, 41])
    plt.xlim([0, 21])
    plt.ylim([-5, 8])
    plt.xlabel('x')
    plt.ylabel('y')

    # Optimize acquisition function
    x_test = [[x] for x in np.linspace(0.0, 20.0, 500)]
    acqopt = AcqOptimizer({'x_test': x_test})
    x_next = acqopt.optimize(model, algo)
    print(f'Acq optimizer x_next = {x_next}')
    print(f'Finished iter i = {i}')

    # Plot true execution path
    plt.plot(exe_path_true.x, exe_path_true.y, '-', color='k', linewidth=3)

    # Show plot
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
