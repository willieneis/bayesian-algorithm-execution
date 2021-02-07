import pickle
import numpy as np
import matplotlib.pyplot as plt

import neatplot
neatplot.set_style('fonts')
neatplot.update_rc('font.size', 20)


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
mes_list, _, _  = parse_method('mes', seed_list)
es_list, _, _  = parse_method('es', seed_list)
_, _, bax_list  = parse_method('bax', seed_list)

results_list = [rs_list, es_list, mes_list, bax_list]
label_list = [
    'Random Search',
    'Evolution Strategy',
    'MV Entropy Search',
    'InfoBAX',
]

#color_list = ["#ff7f0e", "#d62728", "#2ca02c", "#1f77b4"]
#color_list = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728"]
color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"] # Original colors

# Re-scale results given Hartmann6 minimum = -3.322
hartmann_min = -3.3224
results_list = [subtract_min(res, hartmann_min) for res in results_list]

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

h_list = []
for idx, result in enumerate(results_list):
    linecolor = color_list[idx]

    avg_list = np.mean(result, 0)
    stderr_list = np.std(result, 0) / np.sqrt(len(result))
    iters = np.arange(len(avg_list)) + 1
    h = ax.plot(iters, avg_list, '-', color=linecolor)
    h_list.append(h[0])

    lb = avg_list - stderr_list
    ub = avg_list + stderr_list
    ax.fill_between(iters, lb, ub, color=linecolor, alpha=0.1)



# Vertical lines?
ylim = ax.get_ylim()
plt.plot((80, 80), ylim, '--', color="black", alpha=0.2)
plt.plot((500, 500), ylim, '--', color="black", alpha=0.2)

anno_1 = '$t=80$'
ax.annotate(anno_1, (85, 0.035), fontsize=15)
#ax.annotate(anno_1, (50, 0.035))

anno_2 = '$t=500$'
ax.annotate(anno_2, (242, 0.035), fontsize=15)
anno_3 = '(Full $\mathcal{A}$)'
ax.annotate(anno_3, (242, 0.025), fontsize=14)



# Legend
ax.legend(handles=h_list, labels=label_list)
ax.set_yscale('log')
ax.set_xscale('log')

# Lims
ax.set_ylim((0.0189, 3.6))
#ax.set_xlim((1, 550))
#ax.set_xlim((1, 1001))
ax.set_xlim((1, 650))

# Axis labels/titles
ax.set_xlabel('Iteration')
ax.set_ylabel('Simple Regret')
ax.set_title('Hartmann 6 Dimensions')

neatplot.save_figure('hartmann', 'pdf')

plt.show()
