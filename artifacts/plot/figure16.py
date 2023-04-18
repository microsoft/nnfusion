import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
from common import *
import numpy as np
import os

figure_id = 16

sys = ['TorchScript', 'TensorFlow', 'JAX+JIT', sys_name]

colors = [
    color_def[2],
    color_def[3],
    color_def[4]   
]


def get_log_from(filename: str):
    result_dir = f'../reproduce_results/Figure{figure_id}'
    results = []
    for s in ['pytorch', 'tf', 'jax', 'grinder']:
        result_file_path = os.path.join(result_dir, s, filename + '.log')
        t = parse_time(result_file_path)
        if t is not None:
            results.append(t)
        else:
            t = parse_tf_time(result_file_path)
            if t is not None:
                results.append(t)
            else:
                raise ValueError("Cannot parse time from file: {}".format(result_file_path))
    for s in ['pytorch', 'tf', 'jax', 'grinder']:
        result_file_path = os.path.join(result_dir, s, filename + '.nvprof.log')
        t = parse_kernel_time(result_file_path)
        if t is not None:
            results.append(t)
        else:
            raise ValueError("Cannot parse time from file: {}".format(result_file_path))
    return results

blockdrop_base = get_log_from('blockdrop.noskip')
blockdrop_unroll = get_log_from('blockdrop.unroll')
blockdrop_fix = get_log_from('blockdrop.fix')
skipnet_base = get_log_from('skipnet.noskip')
skipnet_unroll = get_log_from('skipnet.unroll')
skipnet_fix = get_log_from('skipnet.fix')

print("blockdrop_base:", blockdrop_base)
print("blockdrop_unroll:", blockdrop_unroll)
print("blockdrop_fix:", blockdrop_fix)
print("skipnet_base:", skipnet_base)
print("skipnet_unroll:", skipnet_unroll)
print("skipnet_fix:", skipnet_fix)

def plot_subfigure(base, unroll, fix, model_name, idx):
    ax = axes[idx]
    vspace = 4
    y = np.arange(4) * vspace
    ax.barh(y, base[:4], edgecolor='k', linewidth=0.1, label='no skip', color=colors[0])
    ax.barh(y + 1, unroll[:4], edgecolor='k', linewidth=0.1, label = 'trace', color=colors[1])
    ax.barh(y + 2, fix[:4], edgecolor='k', linewidth=0.1, label='with control flow', color=colors[2])
    ax.barh(y, base[4:], color='None', hatch='//', edgecolor='k', linewidth=0.1, label='kernel time')
    ax.barh(y + 1, unroll[4:], color='None', hatch='//', edgecolor='k', linewidth=0.1)
    ax.barh(y + 2, fix[4:], color='None', hatch='//', edgecolor='k', linewidth=0.1)
    x_max = fix[0] * 2.5
    
    if idx == 0:
        ax.set_yticks(y + 1)
        ax.set_yticklabels(sys)
    else:
        ax.yaxis.set_major_locator(NullLocator())
    ax.hlines(y[:-1] + 3, xmin=0, xmax=x_max, colors='k', linewidth=0.1)
    ax.set_xlabel('time (ms)', labelpad=-5)
    ax.set_xlim(0, x_max)
    ax.invert_yaxis()
    ax.set_title(model_name, y=-0.3)

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
    # 'figure.constrained_layout.use': True
}
plt.rcParams.update(figsize)
fig = plt.figure(figsize=(8, 3.5))
axes = fig.subplots(nrows=1, ncols=2)

plot_subfigure(blockdrop_base, blockdrop_unroll, blockdrop_fix, '(a) BlockDrop', 0)
plot_subfigure(skipnet_base, skipnet_unroll, skipnet_fix, '(b) SkipNet', 1)

lines, labels = fig.axes[-1].get_legend_handles_labels()
print(labels)

fig.legend(lines, labels, loc = 'upper center', ncol=len(sys), bbox_to_anchor=(0.45, 1.05))
plt.savefig(os.path.join(plot_dir, f'figure{figure_id}.overhead_branch.pdf'), bbox_inches='tight')

print("BlockDrop increase the execution time by")
for i in range(4):
    print(sys[i], "{:.2f}".format(blockdrop_fix[i] / blockdrop_unroll[i] - 1))
print("Blockdrop speedup over base")
for i in range(4):
    print(sys[i], "{:.2f}".format(blockdrop_base[i] / blockdrop_fix[i]))
