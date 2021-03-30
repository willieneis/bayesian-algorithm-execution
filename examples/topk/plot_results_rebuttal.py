import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import neatplot
neatplot.set_style('fonts')
neatplot.update_rc('font.size', 22)


def parse_method(method_str, seed_list):
    eml_list = []

    # Loop through seeds and parse
    for seed in seed_list:

        # Load results
        file_str = f"examples/topk/results_plot/topk_{method_str}_{seed}.pkl"
        results = pickle.load(open(file_str, "rb"))

        eml = results.expected_metric_jacc_list
        #eml = results.expected_metric_norm_list
        eml_list.append(eml)

    return eml_list


def subtract_min(results, f_min):
    results_new = []
    for curve_arr in results:
        curve_arr = curve_arr - f_min
        results_new.append(curve_arr)
    return results_new



seed_list = [1, 2, 3, 4, 5]

# Parse methods' results
eig1_list  = parse_method('eig1', seed_list)
eig2_list = parse_method('eig2', seed_list)
eig3_list  = parse_method('eig3', seed_list)
rand_list  = parse_method('rand', seed_list)
uncert_list  = parse_method('uncert', seed_list)
eigf_list  = parse_method('eigf', [1, 2, 3])

#results_list = [eig1_list, eig2_list, eig3_list, rand_list]
results_list = [rand_list, uncert_list, eig1_list, eig2_list, eig3_list, eigf_list]
label_list = [
    'Random Search',
    'Uncertainty Sampling',
    'EIG$^e_t(x)$, Eq. (4)',
    'EIG$_t(x)$, Eq. (8)',
    'EIG$^v_t(x)$, Eq. (9)',
    'MI$_f$',
    'Top-$k$ algorithm (full)',
]

# reset color cycle
clist = rcParams['axes.prop_cycle']
cgen = itertools.cycle(clist)

# Plot
fig, ax = plt.subplots(figsize=(8, 7))

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

# Top-k algorithm dot
linecolor = "teal"
len_path = 150
h = ax.plot([len_path], [0.0], 'o', markersize=10, color=linecolor)
h_list.append(h[0])


## Vertical lines
ylim = ax.get_ylim()
plt.plot((100, 100), ylim, '--', color="black", alpha=0.2)
plt.plot((150, 150), ylim, '--', color="black", alpha=0.2)


# Legend
ax.legend(handles=h_list, labels=label_list)

#ax.set_yscale('log')
ax.set_xscale('log')

# Add specific ticks and ticklabels
ticks = [10, 50, 100, 150]
ticklabels = ["10", "50", "100", "150"]
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels)

# Lims
ax.set(xlim=(4, 160), ylim=(-0.05, 1.0))

# Axis labels/titles
ax.set_xlabel('Iteration')
ax.set_ylabel('$1 -$ Jaccard Index for Top-$k$')
ax.set_title('Top-$k$ estimation ($k = 10$)')

neatplot.save_figure('topk', 'pdf')

plt.show()
