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
from bax.acq.acquisition_new import (
    BaxAcqFunction, UsBaxAcqFunction, EigfBaxAcqFunction, RandBaxAcqFunction
)
from bax.acq.acqoptimize_new import AcqOptimizer
from bax.util.domain_util import unif_random_sample_domain
from bax.util.graph import make_grid, edges_of_path, positions_of_path

import neatplot
neatplot.set_style('fonts')
neatplot.update_rc('font.size', 20)


seed = 11
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
    label=None,
):
    # plot path taken
    path_lines = edges_of_path(path)
    path_lc = LC(
        path_lines,
        colors=[path_color] * len(path_lines),
        linewidths=linewidths,
        linestyle=linestyle,
        label=label,
    )
    ax.add_collection(path_lc)
    
    # plot visited vertices
    if plot_vertices:
        ax.scatter(*positions_of_path(path).T, color=(0, 0, 0, 1))
    return

def plot_vertices(ax, vertices, **kwargs):
    ax.scatter(*positions_of_path(vertices).T, color=(0, 0, 0, 1), **kwargs)
    return

def plot_acqopt_vertex(ax, vertex, **kwargs):
    ax.scatter(*positions_of_path([vertex]).T, color="blue", **kwargs)
    return

def plot_graph(ax, edges, start, goal):
    # plot edges
    color = (0.75, 0.75, 0.75, 0.1)
    lc = LC(edges, colors=[color] * len(edges), linewidths=1.0)
    ax.add_collection(lc)

    # plot vertices
    ax.scatter(*positions.T, color=(0, 0, 0, 1), marker='.', facecolors='none', s=20)

    # plot start and goal vertices
    ax.scatter(*start.position, color='#9907E1', marker='s', label="Start", s=150)
    ax.scatter(*goal.position, color='#F3C807', marker='s', label="Goal", s=150)
    
    ax.grid(False)
    return

def plot_contourf(fig, ax, x1_lims, x2_lims):
    x, y = np.meshgrid(np.linspace(*x1_lims), np.linspace(*x2_lims))
    
    # plot cost function
    cs = ax.contourf(x, y, rosenbrock(x, y), cmap='BuGn')
    #cbar = fig.colorbar(cs)

# Set up Dijkstra problem
grid_size = 10
x1_lims = (-2, 2)
x2_lims = (-1, 4)
positions, vertices, edges = make_grid(grid_size, x1_lims, x2_lims)
start, goal = vertices[-grid_size], vertices[-1]
algo_params = {
    'cost_func': lambda u, v, f: cost_func(u, v, f, latent_f=True),
    'true_cost': lambda u, v: cost_func(u, v, true_f, latent_f=False)
}
algo = Dijkstra(algo_params, vertices, start, goal)

# Run algorithm on true function
algo_copy = algo.get_copy()
true_ep, true_output = algo_copy.run_algorithm_on_f(true_latent_f)

# Set data for model
n_init = 1
edge_locs = [(e[0] + e[1]) / 2 for e in edges]
x_init_args = np.random.choice(range(len(edge_locs)), n_init)
x_init = [edge_locs[idx] for idx in x_init_args]
data = Namespace()
data.x = [np.array(x).reshape(-1) for x in x_init]
data.y = [true_latent_f(x) for x in data.x]

# Set model details
gp_params = {"ls": 0.3, "alpha": 4.3, "sigma": 1e-2, "n_dimx": 2}
#gp_params = get_stangp_hypers(true_latent_f, domain=[x1_lims, x2_lims], n_samp=400) # NOTE: can use StanGp to fit hypers
modelclass = GpfsGp

# Set method_str here
#method_str = 'bax'
#method_str = 'rand'
#method_str = 'us'
method_str = 'eigf'

print(f'PRINT: method_str = {method_str}')

# Set acquisition details
acqfn_params = {"acq_str": "exe", "n_path": 30}

# Run BAX loop
#n_iter = 70
n_iter = 5

for i in range(n_iter):
    plt.close()
    # Set model
    model = modelclass(gp_params, data)

    # Set and optimize acquisition function
    if method_str == 'bax':
        acqfn = BaxAcqFunction(acqfn_params, model, algo)
    elif method_str == 'rand':
        acqfn = RandBaxAcqFunction(acqfn_params, model, algo)
    elif method_str == 'us':
        acqfn = UsBaxAcqFunction(acqfn_params, model, algo)
    elif method_str == 'eigf':
        acqfn = EigfBaxAcqFunction(acqfn_params, model, algo)

    acqopt = AcqOptimizer({"x_batch": edge_locs, "remove_x_dups": True})
    x_next = acqopt.optimize(acqfn)
    x_next = np.array(x_next).reshape(-1)

    sampled_outputs = acqfn.output_list

    # Check if x_next has been queried before
    if True in [all(x_next == x) for x in data.x]:
        print('\n!!!!!\nThe x_next has already been queried!\n!!!!!\n')

    # Query function, update data
    y_next = true_latent_f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)

    # Print
    print(f'Acq optimizer x_next = {x_next}')
    print(f'Finished iter i = {i}')

    # Plot
    plot = True
    if plot:            
        fig, ax = plt.subplots(figsize=(7.4, 7))

        plot_contourf(fig, ax, x1_lims, x2_lims)

        plot_graph(ax, edges, start, goal)

        plot_path(
            ax,
            true_output[1],
            path_color=(0.2, 0.2, 0.2, 1),
            linewidths=2,
            linestyle="--",
            label='True shortest path',
        )

        for x in data.x[:-1]:
            ax.scatter(x[0], x[1], color=(0, 0, 0, 1), s=120)

        ax.scatter(
            data.x[-1][0],
            data.x[-1][1],
            color='deeppink',
            s=120,
            label='Next query',
        )
       
        ax.set(ylim=[-1.2, 4.2], xlim=[-2.2, 2.2]) # TODO: replace hard coded values
        #ax.legend(loc='lower left')

        min_costs, min_cost_paths = zip(*sampled_outputs)
        weight = 0.1 # NOTE can also do: 1 / acqfn.params.n_path
        for path in min_cost_paths:
            plot_path(ax, path, path_color=(0, 0, 1, weight), linewidths=2, linestyle="-")

        # Plot title
        if method_str == 'bax':
            ax.set_title("InfoBAX")
        elif method_str == 'rand':
            ax.set_title("Random Search")
        elif method_str == 'us':
            ax.set_title("Uncertainty Sampling")
        elif method_str == 'eigf':
            ax.set_title("EIG on $f$")

        # Turn off ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Save
        if method_str == 'bax':
            neatplot.save_figure(f'dij_results/bax_{i}', 'pdf')
        elif method_str == 'rand':
            neatplot.save_figure(f'dij_results/rs_{i}', 'pdf')
        elif method_str == 'us':
            neatplot.save_figure(f'dij_results/us_{i}', 'pdf')
        elif method_str == 'eigf':
            neatplot.save_figure(f'dij_results/eigf_{i}', 'pdf')
