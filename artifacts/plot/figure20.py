from copy import deepcopy
from typing import List
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
from common import *
import os

figure_id = 20

sys =      ['TorchScript', 'TensorFlow', 'JAX+JIT', 'CocktailerBase', sys_name]

hatch_def = [
    '..',
    'xx',
    '//',
    '\\\\',
    '',
    # '++',
    # 'oo',
]

def get_log_from(filename: str):
    result_dir = f'../reproduce_results/Figure{figure_id}'
    results = []
    for s in ['pytorch', 'tf', 'jax', 'base', 'sys']:
        result_file_path = os.path.join(result_dir, s, filename)
        if not os.path.exists(result_file_path):
            results.append('N/A')
        else:
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

lstm_rocm = get_log_from('lstm.b1.log')
nas_rocm = get_log_from('nasrnn.b1.log')
attention_rocm = get_log_from('attention.b1.log')
seq2seq_rocm = get_log_from('seq2seq.b1.log')
blockdrop_rocm = get_log_from('blockdrop.b1.log')
skipnet_rocm = get_log_from('skipnet.b1.log')
rae_rocm = get_log_from('rae.b1.log')

print('lstm_rocm,', ", ".join([str(x) for x in lstm_rocm]))
print('nas_rocm,', ", ".join([str(x) for x in nas_rocm]))
print('attention_rocm,', ", ".join([str(x) for x in attention_rocm]))
print('seq2seq_rocm,', ", ".join([str(x) for x in seq2seq_rocm]))
print('blockdrop_rocm,', ", ".join([str(x) for x in blockdrop_rocm]))
print('skipnet_rocm,', ", ".join([str(x) for x in skipnet_rocm]))
print('rae_rocm,', ", ".join([str(x) for x in rae_rocm]))

def get_second_large(lst: List):
    lst = deepcopy(lst)
    lst.sort()
    return lst[-2]

def plot_subfigure(data, idx, model_name):
    # plt.subfigure(1, 6, idx)
    ax = axes[idx]
    values = [0 if isinstance(x, str) else x for x in data]
    # text = [x if isinstance(x, str) else '' for x in data]
    # x_id = list(range(6))
    for i in range(len(sys)):
        ax.bar([i], [values[i]], label=sys[i], color=color_def[i], hatch=hatch_def[i], edgecolor='black')
    # draw a horizontal line on y=values[-1]
    ax.axhline(y=values[-1], color='red', linestyle='--', linewidth=2)
    best_baseline = list(filter(lambda x: isinstance(x, float), values[:-1]))
    speedup = min(best_baseline) / values[-1]
    ax.text(len(sys) - 1, values[-1], "{:.2f}x".format(speedup), color='black', fontsize=13, horizontalalignment='center',
        verticalalignment='bottom')
    for i, dt in enumerate(data):
        if isinstance(dt, str):
            ax.text(i, 0, dt, rotation='vertical', horizontalalignment='center',
        verticalalignment='bottom')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(NullLocator())
    second_baseline = get_second_large(values[:-2])
    # max_value = max(values)
    max_value = max(second_baseline, values[-2])
    if max(values) < max_value * 3: max_value = max(values)
    # if max(values) < 
    # the second large value in values

    ax.set_ylim((0, max_value * 1.1))
    ax.set_title(model_name, y = -0.3)
    if idx == 0:
        ax.set_ylabel("time (ms)")
        

def avg_speedup(models):
    for i in range(len(sys[:-1])):
        speedups = []
        for model in models:
            if not isinstance(model[i], str):
                speedups.append(model[i] / model[-1])
        print("speedup", sys[i], "mean", "{:.2f}".format(geo_mean(speedups)), "max", "{:.2f}".format(max(speedups)), speedups)

print("Speedup ROCM")
avg_speedup([lstm_rocm, nas_rocm, attention_rocm, seq2seq_rocm, blockdrop_rocm, skipnet_rocm, rae_rocm])

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

fig = plt.figure(figsize=(16, 1.8))
axes = fig.subplots(nrows=1, ncols=7) # sharey=True

plot_subfigure(lstm_rocm, 0, 'LSTM')
plot_subfigure(nas_rocm, 1, 'NASRNN')
plot_subfigure(attention_rocm, 2, 'Attention')
plot_subfigure(seq2seq_rocm, 3, 'Seq2seq')
plot_subfigure(blockdrop_rocm, 4, 'BlockDrop')
plot_subfigure(skipnet_rocm, 5, 'SkipNet')
plot_subfigure(rae_rocm, 6, 'RAE')

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper center', ncol=len(sys), bbox_to_anchor=(0.5, 1.3))
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.ylabel("Exec. time (ms)")
plt.savefig(f"{os.path.join(plot_dir, f'figure{figure_id}.rocm.pdf')}", bbox_inches='tight')
