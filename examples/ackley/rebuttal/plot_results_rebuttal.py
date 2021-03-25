import pickle
import numpy as np
import matplotlib.pyplot as plt

import neatplot
neatplot.set_style('fonts')
neatplot.update_rc('font.size', 22)


def parse_method(method_str, seed_list, dir_id=1):
    bsf_list = []
    mf_list = []
    eo_list = []
    # Loop through seeds and parse
    for seed in seed_list:

        # Load results
        if dir_id == 1:
            file_str = f"examples/ackley/results/{method_str}_{seed}.pkl"
        elif dir_id == 2:
            file_str = f"examples/ackley/rebuttal/results/{method_str}_{seed}.pkl"
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
mes_list, _, _ = parse_method('mes', seed_list)
es_list, _, _  = parse_method('es', seed_list)
_, _, bax_list  = parse_method('bax', seed_list)
_, _, eig1_list  = parse_method('eig1', [11, 12, 13], dir_id=2)
_, _, eig2_list  = parse_method('eig2', [11, 12, 13], dir_id=2)

results_list = [rs_list, es_list, mes_list, eig1_list, eig2_list, bax_list]
label_list = [
    'Random Search',
    'Evolution Strategy',
    'MV Entropy Search',
    'EIG$^e_t(x)$, Eq. (4)',
    'EIG$_t(x)$, Eq. (8)',
    'EIG$^v_t(x)$, Eq. (9)',
]

#color_list = ["#ff7f0e", "#d62728", "#2ca02c", "#1f77b4"]
#color_list = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728"]
#color_list = ["#1f77b4", "#ff7f0e", "#8c564b", "#2ca02c", "#d62728", "#9467bd"] # Original colors
color_list = ["#1f77b4", "#ff7f0e", "#8c564b", "#2ca02c", "#d62728", "#9467bd"] # Original colors

# Plot
fig, ax = plt.subplots(figsize=(8, 7))

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
plt.plot((200, 200), ylim, '--', color="black", alpha=0.2)
plt.plot((2000, 2000), ylim, '--', color="black", alpha=0.2)

anno_1 = '$t=200$'
ax.annotate(anno_1, (215, 2.2), fontsize=15)

anno_2 = '$t=2000$'
ax.annotate(anno_2, (720, 2.2), fontsize=15)
#anno_3 = '(algorithm\n complete)'
#ax.annotate(anno_3, (1200, 0.021))

anno_3 = '(Full $\mathcal{A}$)'
ax.annotate(anno_3, (800, 1.9), fontsize=14)




# Legend
ax.legend(handles=h_list, labels=label_list)
ax.set_yscale('log')
ax.set_xscale('log')

# Lims
#ax.set_ylim((1.8, 16.0))
ax.set_ylim((1.8, 17.0))
#ax.set_xlim((1, 550))
#ax.set_xlim((1, 2700))
ax.set_xlim((1.0, 2700))

# Axis labels/titles
ax.set_xlabel('Iteration')
ax.set_ylabel('Simple Regret')
ax.set_title('Bayesian local optimization ($D = 10$)')

neatplot.save_figure('ackley', 'pdf')

plt.show()
