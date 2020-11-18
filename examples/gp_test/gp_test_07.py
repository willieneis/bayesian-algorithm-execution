import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt

from bax.models.simple_gp import SimpleGp
from bax.alg.algorithms import LinearScanRandGap
from bax.acq.acqoptimize import AcqOptimizer


seed = 11
np.random.seed(seed)

# Set function
f = lambda x: 2 * np.sin(x[0])

# Set model as a GP
model = SimpleGp({'ls': 2.0, 'alpha': 1.5})

# Set data for model
data = Namespace()
data.x = [[1.0], [2.0], [3.0], [10.0]]
data.y = [f(x) for x in data.x]
model.set_data(data)

# Set algorithm
x_path = [[x] for x in np.linspace(3.5, 40, 30)]
algo = LinearScanRandGap({'x_path': x_path})

# Set "true execution path"
x_path_big = [[x] for x in np.linspace(0, 40, 200)]
exe_path_true = Namespace(x=x_path_big, y=[f(x) for x in x_path_big])

# For plotting
fig = plt.figure(figsize=(8, 5))
#plt.xlim([0, 41])
#plt.ylim([-4, 4])
plt.xlabel('x')
plt.ylabel('y')

# Optimize acquisition function
acqopt = AcqOptimizer()
x_next = acqopt.optimize(model, algo)
print(f'Acq optimizer x_next = {x_next}')

# Plot true execution path
plt.plot(exe_path_true.x, exe_path_true.y, '-', color='k', linewidth=3)

plt.show()
