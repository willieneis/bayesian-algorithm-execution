import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import neatplot
neatplot.set_style('fonts')


def parse_method(method_str, seed_list):
    bsf_list = []
    mf_list = []
    eo_list = []
    # Loop through seeds and parse
    for seed in seed_list:

        # Load results
        file_str = f"examples/hartmann/results/{method_str}_{seed}.pkl"
        results = pickle.load(open(file_str, "rb"))

        n_iter = len(results.output_mf_list)

        # Parse data.y and bsf
        y = results.data.y[-n_iter:]
        bsf =  np.minimum.accumulate(y)
        bsf_list.append(bsf)

        # Parse f_output_mf
        mf = results.f_output_mf_list
        mf = np.array(mf)
        mf_list.append(mf)

        # Parse f_expected_output
        eo = results.f_expected_output_list
        eo = np.array(eo)
        eo_list.append(eo)

    return bsf_list, mf_list, eo_list


def subtract_min(results, f_min):
    results_new = []
    for curve_arr in results:
        curve_arr = curve_arr - f_min
        results_new.append(curve_arr)
    return results_new



seed_list = [11, 12, 13, 14, 15]

# Parse methods' results
rs_list, _, _  = parse_method('rs', seed_list)
mes_list, _, _  = parse_method('mes3', seed_list)
es_list, _, _  = parse_method('es', seed_list)
_, _, bax_list  = parse_method('bax2', seed_list)

results_list = [rs_list, es_list, mes_list, bax_list]
label_list = [
    'Random Search',
    'Evolutionary Strategy',
    'Max-value Entropy Search',
    'BAX-Evolutionary',
]

# Re-scale results given Hartmann6 minimum = -3.322
hartmann_min = -3.3224
results_list = [subtract_min(res, hartmann_min) for res in results_list]

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
clist = rcParams['axes.prop_cycle']
cgen = itertools.cycle(clist)

h_list = []
for result in results_list:
    linecolor = next(cgen)['color']

    avg_list = np.mean(result, 0)
    stderr_list = np.std(result, 0) / np.sqrt(len(result))
    iters = np.arange(len(avg_list)) + 1
    h = ax.plot(iters, avg_list, '-', color=linecolor)
    h_list.append(h[0])

    lb = avg_list - stderr_list
    ub = avg_list + stderr_list
    ax.fill_between(iters, lb, ub, color=linecolor, alpha=0.1)

ax.legend(handles=h_list, labels=label_list)
ax.set_yscale('log')
ax.set_xscale('log')

ax.set_ylim((0.0189, 3.6))
ax.set_xlim((1, 550))
#ax.set_xlim((1, 1001))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Hartmann 6 Dimensions')

neatplot.save_figure('hartmann')

plt.show()
