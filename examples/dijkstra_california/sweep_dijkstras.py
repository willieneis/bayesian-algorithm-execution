from pathlib import Path
from argparse import ArgumentParser
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection as LC
import pandas as pd

# plt.ion()
import tensorflow as tf

from bax.alg.dijkstra import Dijkstra

from bax.models.gpfs_gp import GpfsGp
from bax.acq.acquisition_new import BaxAcqFunction, RandBaxAcqFunction, UsBaxAcqFunction
from bax.acq.acqoptimize_new import AcqOptimizer
from bax.util.graph import edges_of_path, positions_of_path, area_of_polygons
from bax.util.graph import make_vertices, make_edges, connected_components

import neatplot
import pickle

neatplot.set_style("fonts")

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--n_init", type=int, default=1, help="Number of initial queries")
parser.add_argument("--n_iter", type=int, default=70, help="Number of queries to make")
parser.add_argument("--plot", action="store_true", help="Whether to plot")
parser.add_argument(
    "--n_path",
    type=int,
    default=20,
    help="Number of posterior samples to draw at each iteration",
)
parser.add_argument(
    "--can_requery",
    action="store_true",
    help="Whether or not the acquisition function can query a previously queried point",
)
parser.add_argument(
    "--acq_func",
    type=str,
    choices=["bax", "rand", "uncert"],
    help="Type of acquisition function. Choose one of BAX, random search, and uncertainty sampling.",
)
parser.add_argument("--save_dir", type=str, default="./")
args = parser.parse_args()

exp_dir = Path(args.save_dir) / args.acq_func / f"seed_{args.seed}"
exp_dir.mkdir(parents=True, exist_ok=True)

with open(exp_dir / "settings.pkl", "wb") as handle:
    pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)

seed = args.seed
np.random.seed(seed)
tf.random.set_seed(seed)

df_edges = pd.read_csv("data/ba_edges_new.csv")
df_nodes = pd.read_csv("data/ba_nodes_new.csv")

df_edges["elevation"] = (df_edges["elevation_y"] + df_edges["elevation_x"]) / 2
edge_elevation = df_edges["elevation"].to_numpy()

# rescaling
node_elevations = df_nodes["elevation"] / edge_elevation.max() + 0.1

edge_elevation = edge_elevation / edge_elevation.max()  # scale between [0, 1]
edge_elevation = (
    edge_elevation + 0.1
)  # make strictly positive to prevent inv_softmax blowup
edge_positions = df_edges[["mean_longitude", "mean_latitude"]].to_numpy()


def normalize(data, scale):
    data = data - data.min(0, keepdims=True)
    return data / scale


# TODO: check lims
x1_lims = (-123, -119)
x2_lims = (36.8, 39.1)
# TODO: rescale data x1,x2 and y


# normalize both x and y by longitude
xy_normalization = edge_positions[:, 0].max() - edge_positions[:, 0].min()
edge_positions = normalize(edge_positions, xy_normalization)
print(xy_normalization)

edge_tuples = [tuple(e) for e in df_edges[["start_nodeid", "end_nodeid"]].to_numpy()]
edge_tuples = edge_tuples + [(v, u) for (u, v) in edge_tuples]
# make undirected
edge_to_elevation = dict(
    zip(
        edge_tuples,
        np.concatenate([edge_elevation, edge_elevation]),
    )
)
edge_to_position = dict(
    zip(
        edge_tuples,
        np.concatenate([edge_positions, edge_positions]),
    )
)
edge_position_to_elevation = dict(
    zip([tuple(p) for p in edge_positions], edge_elevation)
)
# right now we use the original node elevation to plot nodes but run experiment
# using scaled elevation
positions = df_nodes[["longitude", "latitude"]].to_numpy()
positions = normalize(positions, xy_normalization)
edge_nodes = df_edges[["start_nodeid", "end_nodeid"]].to_numpy()

has_edge = np.zeros((len(positions), len(positions)))
# make undirected edges
has_edge[edge_nodes[:, 0], edge_nodes[:, 1]] = 1
has_edge[edge_nodes[:, 1], edge_nodes[:, 0]] = 1

# Set function
from bax.models.gp.gp_utils import (
    get_cholesky_decomp,
    solve_upper_triangular,
    solve_lower_triangular,
    kern_exp_quad,
)


def gp_post_mean(
    x_train,
    y_train,
    x_pred,
    ls,
    alpha,
    sigma,
    kernel=kern_exp_quad,
    smat=None,
    **kwargs,
):
    """Compute parameters of GP posterior"""
    if smat is None:
        k11_nonoise = kernel(x_train, x_train, ls, alpha)
        lmat = get_cholesky_decomp(k11_nonoise, sigma, "try_first")
        smat = solve_upper_triangular(lmat.T, solve_lower_triangular(lmat, y_train))
    k21 = kernel(x_pred, x_train, ls, alpha)
    mu2 = k21.dot(smat)
    return mu2, smat


# def rosenbrock(x, y, a=1, b=100):
#    # NOTE rescaled to improve numerics
#    return 1e-2 * ((a - x) ** 2 + b * (y - x ** 2) ** 2)
#
#
# def true_f(x_y):
#    x_y = np.array(x_y).reshape(-1)
#    return rosenbrock(x_y[..., 0], x_y[..., 1])

# gp_params = get_stangp_hypers_xy(
#    list(edge_positions), list(edge_elevation), domain=[x1_lims, x2_lims]
# )  # NOTE: can use StanGp to fit hypers
gp_params = {
    "ls": 0.07768287838990902,
    "alpha": 0.2007688897833724,
    "sigma": 0.01,
    "n_dimx": 2,
}
# edge_data = Namespace()
# edge_data.x = list(edge_positions)
# edge_data.y = list(edge_elevation)

_, smat = gp_post_mean(
    edge_positions, edge_elevation, np.array([[1.0, 1.0]]), **gp_params
)


def true_f(x):
    assert x.ndim == 1
    # mu, _ = gp_post_mean(
    #    edge_positions, edge_elevation, x.reshape(1, -1), smat=smat, **gp_params
    # )

    # return mu[0]
    return edge_position_to_elevation[tuple(x)]


def inv_softplus(x):
    return x + np.log(-np.expm1(-x))  # numerical stability
    # return np.log(np.exp(x) - 1)


# NOTE: this is the function we will use
def true_latent_f(x_y):
    return inv_softplus(true_f(x_y))


# Set other useful functions


def softplus(x):
    return np.log1p(np.exp(x))


def cost_func(u, v, f, latent_f=True):
    edge = (u.index, v.index)
    edge_pos = edge_to_position[edge]
    edge_cost = f(edge_pos)
    if latent_f:
        return softplus(edge_cost), [edge_pos], [edge_cost]
    else:
        return edge_cost, [edge_pos], [edge_cost]


def true_cost_func(u, v):
    edge = (u.index, v.index)
    edge_cost = edge_to_elevation[edge]
    return edge_cost, [edge], [edge_cost]


def cost_of_path(path, cost_func):
    cost = 0
    for i in range(len(path) - 1):
        cost += cost_func(path[i], path[i + 1])[0]  # index to get edge_cost
    return cost


def plot_path(
    ax,
    path,
    path_color=(0, 0, 0, 1.0),
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
    # ax.scatter(*positions.T, color=(0, 0, 0, 1), facecolors="none")
    ax.scatter(*positions.T, c=node_elevations, cmap="BuGn")

    # plot start and goal vertices
    ax.scatter(*start.position, color="#9907E1", marker="s", label="Start", s=130)
    ax.scatter(*goal.position, color="#F3C807", marker="s", label="Goal", s=130)

    ax.grid(False)
    return


# def plot_contourf(fig, ax, x1_lims, x2_lims):
#    x, y = np.meshgrid(np.linspace(*x1_lims), np.linspace(*x2_lims))
#
#    # plot cost function
#    cs = ax.contourf(x, y, rosenbrock(x, y), cmap="BuGn")
#    cbar = fig.colorbar(cs)


def true_cost_of_path(path):
    # cf = lambda u, v: cost_func(u, v, true_f, latent_f=False)
    return cost_of_path(path, true_cost_func)


# Set up Dijkstra problem

algo_params = {
    "cost_func": lambda u, v, f: cost_func(u, v, f, latent_f=True),
    "true_cost": lambda u, v: true_cost_func(u, v),
    # "true_cost": lambda u, v: cost_func(u, v, true_f, latent_f=False),
}

vertices = make_vertices(positions, has_edge)
edges = make_edges(vertices)
start = vertices[3939]  # ~ Santa Cruz
goal = vertices[446]  # ~ Lake Tahoe
components = connected_components(vertices)
components = [[v.index for v in c] for c in components]
# make sure start and goal are in the same connected component
assert any(all((start.index in c, goal.index in c)) for c in components)

algo = Dijkstra(algo_params, vertices, start, goal, edge_to_position)

# Set model details
modelclass = GpfsGp

# Run algorithm on true function
algo_copy = algo.get_copy()
true_ep, true_output = algo_copy.run_algorithm_on_f(true_latent_f)

# Set data for model
n_init = args.n_init
# edge_locs = [(e[0] + e[1]) / 2 for e in edges]
edge_locs = list(edge_positions)
x_init_args = np.random.choice(range(len(edge_locs)), n_init)
x_init = [edge_positions[idx] for idx in x_init_args]
data = Namespace()
data.x = [np.array(x).reshape(-1) for x in x_init]
data.y = [true_latent_f(x) for x in data.x]

# Set acquisition details
acqfn_params = {"acq_str": "exe", "n_path": args.n_path}
# n_rand_acqopt = 350

# Run BAX loop
n_iter = args.n_iter
plot = args.plot

true_costs = []
areas = []

for i in range(n_iter):
    plt.close()
    # Set model
    model = modelclass(gp_params, data)

    # Set and optimize acquisition function
    acqfn_cls = {
        "bax": BaxAcqFunction,
        "rand": RandBaxAcqFunction,
        "uncert": UsBaxAcqFunction,
    }
    acqfn = acqfn_cls[args.acq_func](acqfn_params, model, algo)
    acqopt = AcqOptimizer({"x_batch": edge_locs, "remove_x_dups": args.can_requery})
    x_next = acqopt.optimize(acqfn)
    x_next = np.array(x_next).reshape(-1)

    sampled_outputs = acqfn.output_list

    # Check if x_next has been queried before
    if not args.can_requery and True in [all(x_next == x) for x in data.x]:
        print("\n!!!!!\nThe x_next has already been queried!\n!!!!!\n")

    # Query function, update data
    y_next = true_latent_f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)

    # Print
    print(f"Acq optimizer x_next = {x_next}")
    print(f"Finished iter i = {i}")

    min_costs, min_cost_paths = zip(*sampled_outputs)
    true_costs_of_sampled_paths = [true_cost_of_path(p) for p in min_cost_paths]
    true_costs.append(true_costs_of_sampled_paths)
    np.save(exp_dir / "true_costs.npy", np.array(true_costs))

    grid_area = (x1_lims[1] - x1_lims[0]) * (x2_lims[1] - x2_lims[0])
    areas_between_paths = [
        area_of_polygons(path, true_output[1]) / grid_area for path in min_cost_paths
    ]
    areas.append(areas_between_paths)
    np.save(exp_dir / "areas.npy", np.array(areas))

    # Plot
    if plot:
        fig, ax = plt.subplots(figsize=(7, 7))

        # plot_contourf(fig, ax, x1_lims, x2_lims)

        plot_graph(ax, edges, start, goal)

        plot_path(
            ax,
            true_output[1],
            path_color=(0.2, 0.2, 0.2, 1),
            linewidths=2,
            linestyle="--",
            label="True shortest path",
        )

        for x in data.x[1:-1]:  # skip initial random
            ax.scatter(x[0], x[1], color=(0, 0, 0, 1))

        ax.scatter(
            data.x[-1][0],
            data.x[-1][1],
            color="deeppink",
            s=120,
            label="Next query",
        )

        # ax.set(ylim=[-1.2, 4.2], xlim=[-2.2, 2.2])  # TODO: replace hard coded values
        # ax.set(
        #    ylim=[x2_lims[0] - 0.2, x2_lims[1] + 0.2],
        #    xlim=[x1_lims[0] - 0.2, x1_lims[1] + 0.2],
        # )
        # ax.legend(loc="lower left")
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        weight = 0.1  # NOTE can also do: 1 / acqfn.params.n_path
        for path in min_cost_paths:
            plot_path(
                ax, path, path_color=(0, 0, 1, weight), linewidths=2, linestyle="-"
            )

        method_title = {
            "bax": "InfoBAX",
            "rand": "Random Search",
            "uncert": "Uncertainty Sampling",
        }
        ax.set_title(method_title[args.acq_func])

        # Pause
        # inp = input("Press enter to continue (any other key to stop): ")
        # if inp:
        # break
        # plt.close()

        # make matplotlib plot within for loop. See: https://stackoverflow.com/questions/19766100/real-time-matplotlib-plot-is-not-working-while-still-in-a-loop
        # plt.show()

        neatplot.save_figure((exp_dir / f"bax_{i}").as_posix())
        # plt.pause(0.0001)
        # inp = input("Paused")
