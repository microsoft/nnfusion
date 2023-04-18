import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
from common import *
import numpy as np
import os

figure_id = 15

sys = ['TorchScript', 'TensorFlow', 'JAX+JIT', sys_name]

colors = [
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


lstm_unroll = get_log_from('lstm.unroll')
lstm_fix = get_log_from('lstm.fix')
nasrnn_unroll = get_log_from('nasrnn.unroll')
nasrnn_fix = get_log_from('nasrnn.fix')
attention_unroll = get_log_from('attention.unroll')
attention_fix = get_log_from('attention.fix')
seq2seq_unroll = get_log_from('seq2seq.unroll')
seq2seq_fix = get_log_from('seq2seq.fix')

print('lstm_unroll:', lstm_unroll)
print('lstm_fix:', lstm_fix)
print('nasrnn_unroll:', nasrnn_unroll)
print('nasrnn_fix:', nasrnn_fix)
print('attention_unroll:', attention_unroll)
print('attention_fix:', attention_fix)
print('seq2seq_unroll:', seq2seq_unroll)
print('seq2seq_fix:', seq2seq_fix)


def plot_subfigure(unroll, fix, model_name, row_id, col_id):
    ax = axes[row_id][col_id]
    vspace = 3
    y = np.arange(4) * vspace
    ax.barh(y, unroll[:4], edgecolor='k', linewidth=0.1, label = 'trace', color=colors[0])
    ax.barh(y + 1, fix[:4], edgecolor='k', linewidth=0.1, label='with control flow', color=colors[1])
    ax.barh(y, unroll[4:], color='None', hatch='//', edgecolor='k', linewidth=0.1,  label='kernel time')
    ax.barh(y + 1, fix[4:], color='None', hatch='//', edgecolor='k', linewidth=0.1)
    x_max = fix[0] * 1.5
    
    if col_id == 0:
        ax.set_yticks(y + 1)
        ax.set_yticklabels(sys)
    else:
        ax.yaxis.set_major_locator(NullLocator())
    ax.hlines(y[:-1] + 2, xmin=0, xmax=x_max, colors='k', linewidth=0.1)
    ax.set_xlabel('time (ms)', labelpad=-5)
    ax.set_xlim(0, x_max)
    ax.invert_yaxis()
    ax.set_title(model_name, y=-0.45)

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
}
plt.rcParams.update(figsize)
fig = plt.figure(figsize=(8, 6))
fig.subplots_adjust(hspace=0.5)
axes = fig.subplots(nrows=2, ncols=2)

plot_subfigure(lstm_unroll, lstm_fix, '(a) LSTM', 0, 0)
plot_subfigure(nasrnn_unroll, nasrnn_fix, '(b) NASRNN', 0, 1)
plot_subfigure(attention_unroll, attention_fix, '(c) Attention', 1, 0)
plot_subfigure(seq2seq_unroll, seq2seq_fix, '(d) Seq2seq', 1, 1)

lines, labels = fig.axes[-1].get_legend_handles_labels()
print(labels)

fig.legend(lines, labels, loc = 'upper center', ncol=len(sys), bbox_to_anchor=(0.5, 0.97))
plt.savefig(os.path.join(plot_dir, f'figure{figure_id}.overhead_loop.pdf'), bbox_inches='tight')
