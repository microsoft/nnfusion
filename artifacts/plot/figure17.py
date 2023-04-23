import linecache
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator, FixedLocator
from common import *
import numpy as np
import os

figure_id = 17
sys = ['TorchScript', 'TensorFlow', 'JAX+JIT', sys_name]
# sys = ['JAX+JIT', sys_name]

colors = [line_colors[0], line_colors[1], line_colors[2], line_colors[4]]

def get_log_from(filename: str):
    result_dir = f'../reproduce_results/Figure{figure_id}'
    results = [-1, -1] # skip pytorch and tf
    for s in ['jax', 'sys']:
        result_file_path = os.path.join(result_dir, s, filename)
        t = parse_time(result_file_path)
        if t is not None:
            results.append(t)
        else:
            t = parse_tf_time(result_file_path)
            if t is not None:
                results.append(t)
            else:
                raise ValueError("Cannot parse time from file: {}".format(result_file_path))

    return results

def plot_subfigure(unroll, fix, base, model_name, idx):
    ax = axes[idx]
    rate = [0, 25, 50, 75, 100]
    for i, sys_name in enumerate(sys):
        if i == 0 or i == 1: continue
        print(rate, unroll[:5, i])
        ax.plot(rate, [base[i]] * len(rate), label=sys_name + ' (no skip)', color=colors[i], linestyle=line_styles[0])
        ax.plot(rate, unroll[:5, i], label=sys_name + " (trace)", linestyle=line_styles[1], color=colors[i], marker=line_markers[i])
        ax.plot(rate, fix[:5, i], label=sys_name + "", linestyle=line_styles[2], color=colors[i], marker = line_markers[i])
    ax.set_title(model_name, y = -0.5)
    ax.xaxis.set_major_locator(FixedLocator(rate))
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_xlabel('executed ratio')
    if idx == 0:
        ax.set_ylabel("time (ms)")



blockdrop_fix = []
blockdrop_unroll = []
skipnet_fix = []
skipnet_unroll = []
for i in [0, 25, 50, 75, 100]:
    blockdrop_fix.append(get_log_from(f'blockdrop.{i}.fix.log'))
    blockdrop_unroll.append(get_log_from(f'blockdrop.{i}.unroll.log'))
blockdrop_base = get_log_from(f'blockdrop.noskip.log')

for i in [0, 25, 50, 75, 100]:
    skipnet_fix.append(get_log_from(f'skipnet.{i}.fix.log'))
    skipnet_unroll.append(get_log_from(f'skipnet.{i}.unroll.log'))
skipnet_base = get_log_from(f'skipnet.noskip.log')

blockdrop_fix = np.array(blockdrop_fix)
blockdrop_unroll = np.array(blockdrop_unroll)
skipnet_fix = np.array(skipnet_fix)
skipnet_unroll = np.array(skipnet_unroll)

print('blockdrop_base', blockdrop_base)
print('blockdrop_unroll', blockdrop_unroll)
print('blockdrop_fix', blockdrop_fix)
print('skipnet_base', skipnet_base)
print('skipnet_unroll', skipnet_unroll)
print('skipnet_fix', skipnet_fix)

print("rate 0 speedup", "blockdrop", blockdrop_fix[0, 2] / blockdrop_fix[0, 3], "skipnet", skipnet_fix[0, 2] / skipnet_fix[0, 3])

figsize = {
    # "figure.figsize": (16, 2),
    'font.sans-serif': 'Times New Roman',
    'axes.labelsize': 18,
    'font.size': 18,
    'legend.fontsize': 15,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.constrained_layout.use': True
}
plt.rcParams.update(figsize)
fig = plt.figure(figsize=(8, 3))
# fig = plt.figure(figsize=(6, 2))
axes = fig.subplots(nrows=1, ncols=2)
plot_subfigure(blockdrop_unroll, blockdrop_fix, blockdrop_base, '(a) BlockDrop', 0)
plot_subfigure(skipnet_unroll, skipnet_fix, skipnet_base, '(b) SkipNet', 1)

lines, labels = fig.axes[-1].get_legend_handles_labels()
lines = [lines[0], lines[3], lines[1], lines[4], lines[2], lines[5]]
labels = [labels[0], labels[3], labels[1], labels[4], labels[2], labels[5]]
fig.legend(lines, labels, loc = 'upper center', ncol=3, bbox_to_anchor=(0.5, 1.3))
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.savefig(f"{os.path.join(plot_dir, f'figure{figure_id}.skiprate.pdf')}", bbox_inches='tight')