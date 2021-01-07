import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import tensorflow as tf

from bax.alg.dijkstra import Dijkstra
from bax.util.graph import make_grid


seed = 12
np.random.seed(seed)
tf.random.set_seed(seed)


# Set function
def rosenbrock(x, y, a=1, b=100):
    # NOTE rescaled to improve numerics
    # NOTE min cost path: 1.0527267184880365
    return 1e-2 * ((a - x)**2 + b * (y - x**2)**2)

def true_f(x_y):
    return rosenbrock(x_y[..., 0], x_y[..., 1])

def inv_softplus(x):
        return np.log(np.exp(x) - 1)

def true_latent_f(x_y):
    return inv_softplus(true_f(x_y))

def softplus(x):
        return np.log1p(np.exp(x))

def cost_func(u, v, f, latent_f=True):
    u_pos, v_pos = u.position, v.position
    edge = (u_pos + v_pos) / 2
    edge_cost = f(edge)
    if latent_f:
        return softplus(edge_cost), [edge], [edge_cost]
    else:
        return edge_cost, [edge], [edge_cost]


# Set up Dijkstra problem
grid_size = 10
x1_lims = (-2, 2)
x2_lims = (-1, 4)
positions, vertices, edges = make_grid(grid_size, x1_lims, x2_lims)

start, goal = vertices[-grid_size], vertices[-1]

algo_params = {
    'start': start,
    'goal': goal,
    'vertices': vertices,
    'cost_func': lambda u, v, f: cost_func(u, v, f, latent_f=True),
    'true_cost': lambda u, v: cost_func(u, v, true_f, latent_f=False)
}
algo1 = Dijkstra(algo_params)

a1, b1 = algo1.run_algorithm_on_f(true_latent_f)


algo = Dijkstra(algo_params)
algo.initialize()

x = 1
while True:
    x = algo.get_next_x()
    if x is None:
        break
    y = true_latent_f(x)
    algo.exe_path.x.append(x)
    algo.exe_path.y.append(y)

a = algo.exe_path
b = algo.get_output()
