from copy import deepcopy
from typing import List
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
from common import *
import os

figure_id = 14

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

lstm_1 = get_log_from('lstm.b1.log')
lstm_64 = get_log_from('lstm.b64.log')
nas_1 = get_log_from('nasrnn.b1.log')
nas_64 = get_log_from('nasrnn.b64.log')
attention_1 = get_log_from('attention.b1.log')
attention_64 = get_log_from('attention.b64.log')
seq2seq_1 = get_log_from('seq2seq.b1.log')
seq2seq_64 = get_log_from('seq2seq.b64.log')
blockdrop_1 = get_log_from('blockdrop.b1.log')
blockdrop_64 = get_log_from('blockdrop.b64.log')
skipnet_1 = get_log_from('skipnet.b1.log')
rae_1 = get_log_from('rae.b1.log')

print("LSTM 1", lstm_1)
print("LSTM 64", lstm_64)
print("NAS 1", nas_1)
print("NAS 64", nas_64)
print("Attention 1", attention_1)
print("Attention 64", attention_64)
print("Seq2Seq 1", seq2seq_1)
print("Seq2Seq 64", seq2seq_64)
print("BlockDrop 1", blockdrop_1)
print("BlockDrop 64", blockdrop_64)
print("SkipNet 1", skipnet_1)
print("RAE 1", rae_1)

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

def best_avg_speedup(models):
    speedups = []
    for model in models:
        best_baseline = list(filter(lambda x: isinstance(x, float), model[:-1]))
        speedups.append(min(best_baseline) / model[-1])
    print("best speedup", "mean", "{:.2f}".format(geo_mean(speedups)), "max", "{:.2f}".format(max(speedups)), speedups)

print("Speedup NVIDIA")
avg_speedup([lstm_1, lstm_64, nas_1, nas_64, attention_1, attention_64, seq2seq_1, seq2seq_64, blockdrop_1, blockdrop_64, skipnet_1, rae_1])
print("Best avg speedup")
best_avg_speedup([lstm_1, lstm_64, nas_1, nas_64, attention_1, attention_64, seq2seq_1, seq2seq_64, blockdrop_1, blockdrop_64, skipnet_1, rae_1])

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
# fig.constrained_layout()
axes = fig.subplots(nrows=1, ncols=7) # sharey=True

plot_subfigure(lstm_1, 0, 'LSTM')
plot_subfigure(nas_1, 1, 'NASRNN')
plot_subfigure(attention_1, 2, 'Attention')
plot_subfigure(seq2seq_1, 3, 'Seq2seq')
plot_subfigure(blockdrop_1, 4, 'BlockDrop')
plot_subfigure(skipnet_1, 5, 'SkipNet')
plot_subfigure(rae_1, 6, 'RAE')

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper center', ncol=len(sys), bbox_to_anchor=(0.5, 1.3))
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.savefig(f"{os.path.join(plot_dir, f'figure{figure_id}.bs1.pdf')}", bbox_inches='tight')

fig = plt.figure(figsize=(16, 1.8))
axes = fig.subplots(nrows=1, ncols=5) # sharey=True

plot_subfigure(lstm_64, 0, 'LSTM')
plot_subfigure(nas_64, 1, 'NASRNN')
plot_subfigure(attention_64, 2, 'Attention')
plot_subfigure(seq2seq_64, 3, 'Seq2seq')
plot_subfigure(blockdrop_64, 4, 'BlockDrop')

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.savefig(f"{os.path.join(plot_dir, f'figure{figure_id}.bs64.pdf')}", bbox_inches='tight')
if SHOW: plt.show()
