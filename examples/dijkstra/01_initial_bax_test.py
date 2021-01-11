import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection as LC
#plt.ion()
import tensorflow as tf

from bax.alg.dijkstra import Dijkstra
from bax.models.simple_gp import SimpleGp
from bax.models.gpfs_gp import GpfsGp
from bax.models.stan_gp import get_stangp_hypers
from bax.acq.acquisition_new import BaxAcqFunction
from bax.acq.acqoptimize_new import AcqOptimizer
from bax.util.graph import make_grid, edges_of_path, positions_of_path

import neatplot
neatplot.set_style('fonts')


seed = 12
np.random.seed(seed)
tf.random.set_seed(seed)


# Set function
def rosenbrock(x, y, a=1, b=100):
    # NOTE rescaled to improve numerics
    # NOTE min cost path: 1.0527267184880365
    return 1e-2 * ((a - x)**2 + b * (y - x**2)**2)

def true_f(x_y):
    x_y = np.array(x_y).reshape(-1)
    return rosenbrock(x_y[..., 0], x_y[..., 1])

def inv_softplus(x):
        return np.log(np.exp(x) - 1)

# NOTE: this is the function we will use
def true_latent_f(x_y):
    return inv_softplus(true_f(x_y))


# Set other useful functions

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

def cost_of_path(path, cost_func):
    cost = 0
    for i in range(len(path) - 1):
        cost += cost_func(path[i], path[i+1])
    return cost

def plot_path(
    ax,
    path,
    path_color=(0, 0, 0, 1.),
    linewidths=2,
    linestyle="dotted",
    plot_vertices=False,
):
    # plot path taken
    path_lines = edges_of_path(path)
    path_lc = LC(
        path_lines,
        colors=[path_color] * len(path_lines),
        linewidths=linewidths,
        linestyle=linestyle,
    )
    ax.add_collection(path_lc)
    
    # plot visited vertices
    if plot_vertices:
        ax.scatter(*positions_of_path(path).T, color=(0, 0, 0, 1))
    return

def plot_vertices(ax, vertices, **kwargs):
    ax.scatter(*positions_of_path(vertices).T, color=(0, 0, 0, 1), **kwargs)
    return

def plot_graph(ax, edges, start, goal):
    # plot edges
    color = (0.75, 0.75, 0.75, 0.1)
    lc = LC(edges, colors=[color] * len(edges), linewidths=1.0)
    ax.add_collection(lc)

    # plot vertices
    ax.scatter(*positions.T, color=(0, 0, 0, 1), facecolors='none')

    # plot start and goal vertices
    ax.scatter(*start.position, color='g', label="Start", s=100)
    ax.scatter(*goal.position, color='r', label="Goal", s=100)
    
    ax.grid(False)
    ax.legend()
    return

def plot_contourf(fig, ax, x1_lims, x2_lims):
    x, y = np.meshgrid(np.linspace(*x1_lims), np.linspace(*x2_lims))
    
    # plot cost function
    cs = ax.contourf(x, y, rosenbrock(x, y), cmap='BuGn')
    cbar = fig.colorbar(cs)

# Set up Dijkstra problem
grid_size = 20
x1_lims = (-2, 2)
x2_lims = (-1, 4)
positions, vertices, edges = make_grid(grid_size, x1_lims, x2_lims)
start, goal = vertices[-grid_size], vertices[-1]
algo_params = {
    'cost_func': lambda u, v, f: cost_func(u, v, f, latent_f=True),
    'true_cost': lambda u, v: cost_func(u, v, true_f, latent_f=False)
}
algo = Dijkstra(algo_params, vertices, start, goal)

# Set data for model
data = Namespace()
data.x = [start.position]
data.y = [true_latent_f(x) for x in data.x]

# Set model details
gp_params = {"ls": 0.3, "alpha": 4.3, "sigma": 1e-2, "n_dimx": 2}
#gp_params = {"ls": 0.75, "alpha": 4.3, "sigma": 1e-2, "n_dimx": 2}
#gp_params = get_stangp_hypers(true_latent_f, domain=[x1_lims, x2_lims], n_samp=400) # NOTE: can use StanGp to fit hypers
modelclass = GpfsGp
#modelclass = SimpleGp # NOTE: can use SimpleGp model

# Set acquisition details
acqfn_params = {"acq_str": "exe", "n_path": 20}
#acqfn_params = {"acq_str": "out", "n_path": 20} # TODO: implement "out" acqfn 
n_rand_acqopt = 350

# Run BAX loop
n_iter = 25

queried_xs = []
for i in range(n_iter):
    # Set model
    model = modelclass(gp_params, data)

    # Set and optimize acquisition function
    acqfn = BaxAcqFunction(acqfn_params, model, algo)
    acqopt = AcqOptimizer({"x_batch": positions})
    x_next = acqopt.optimize(acqfn)
    arg_x_next, _ = next(
        (i for i in enumerate(vertices) if all(i[1].position == x_next))
    )

    sampled_outputs = acqfn.output_list

    # Query function, update data
    y_next = true_latent_f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)

    # Update physical path taken
    next_vertex = vertices[arg_x_next]
    queried_xs.append(next_vertex)

    # Print
    print(f'Acq optimizer x_next = {next_vertex}')
    print(f'With x_next.position = {next_vertex.position}')
    print(f'Finished iter i = {i}')

    # Plot
    plot = True
    if plot:            
        fig, ax = plt.subplots(figsize=(9, 7))

        plot_contourf(fig, ax, x1_lims, x2_lims)

        plot_graph(ax, edges, start, goal)

        if len(queried_xs[:-1]) > 0:
            plot_vertices(ax, queried_xs[:-1], label='Given')
       
        ax.set(ylim=[-1.2, 4.2], xlim=[-2.2, 2.2]) # TODO: replace hard coded values
        ax.legend()

        min_costs, min_cost_paths = zip(*sampled_outputs)
        n = acqfn.params.n_path
        for path in min_cost_paths:
            plot_path(ax, path, path_color=(0, 0, 1, 1/n), linewidths=2, linestyle="-")

        # make matplotlib plot within for loop. See: https://stackoverflow.com/questions/19766100/real-time-matplotlib-plot-is-not-working-while-still-in-a-loop
        plt.show()

        # Pause
        inp = input("Press enter to continue (any other key to stop): ")
        if inp:
            break
        plt.close()
