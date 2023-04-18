import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
from common import *
import numpy as np
import os

figure_id = 18

sys = ['PyTorch', 'JAX', sys_name]

colors = [
    color_def[3],
    color_def[4]   
]


def get_log_from(filename: str):
    result_dir = f'../reproduce_results/Figure{figure_id}'
    results = []
    for s in ['pytorch', 'jax', 'grinder']:
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
    for s in ['pytorch', 'jax', 'grinder']:
        result_file_path = os.path.join(result_dir, s, filename + '.nvprof.log')
        t = parse_kernel_time(result_file_path)
        if t is not None:
            results.append(t)
        else:
            raise ValueError("Cannot parse time from file: {}".format(result_file_path))
    return results


rae_fix = get_log_from('rae.fix')
rae_unroll = get_log_from('rae.unroll')
print('rae_fix:', rae_fix)
print('rae_unroll:', rae_unroll)

print("sys time increase over unroll", rae_fix[-1] / rae_unroll[-1])

def plot_subfigure(unroll, fix, idx):
    ax = axes
    vspace = 3
    y = np.arange(3) * vspace
    ax.barh(y, unroll[:3], edgecolor='k', linewidth=0.1, label = 'trace', color=colors[0])
    ax.barh(y + 1, fix[:3], edgecolor='k', linewidth=0.1, label='with control flow', color=colors[1])
    ax.barh(y, unroll[3:], color='None', hatch='//', edgecolor='k', linewidth=0.1, label='kernel time')
    ax.barh(y + 1, fix[3:], color='None', hatch='//', edgecolor='k', linewidth=0.1)
    x_max = fix[0] * 1.5
    
    if idx == 0:
        ax.set_yticks(y + 1)
        ax.set_yticklabels(sys)
    else:
        ax.yaxis.set_major_locator(NullLocator())
    ax.hlines(y[:-1] + 2, xmin=0, xmax=x_max, colors='k', linewidth=0.1)
    ax.set_xlabel('time (ms)', labelpad=0.1)
    ax.set_xlim(0, x_max)
    ax.invert_yaxis()

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
fig = plt.figure(figsize=(8, 2))
axes = fig.subplots(nrows=1, ncols=1)

plot_subfigure(rae_unroll, rae_fix, 0)

lines, labels = fig.axes[-1].get_legend_handles_labels()
print(labels)

fig.legend(lines, labels, loc = 'upper center', ncol=len(sys), bbox_to_anchor=(0.5, 1.25))
# if SHOW: plt.show()
plt.savefig(os.path.join(plot_dir, f'figure{figure_id}.overhead_recursion.pdf'), bbox_inches='tight')

