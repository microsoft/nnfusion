import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
import matplotlib.gridspec as gridspec
from common import *
import os

figure_id = 19

color_general = [color_def[2], color_def[3], color_def[4]]
hatch_general = [
    '//',
    '\\\\',
    '',
]

color_rae = [color_def[0], color_def[1], color_def[2], color_def[3], color_def[4]]
hatch_rae = [
    '.',
    'xx',
    '//',
    '\\\\',
    '',
]

sys_general = ['CocktailerBase', 'schedule', 'optimize & schedule']

def get_log_from(filename: str):
    result_dir = f'../reproduce_results/Figure{figure_id}'
    results = []
    for s in ['base', 'schedule', 'sys']:
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


lstm = get_log_from('lstm.b1.log')
nasrnn = get_log_from('nasrnn.b1.log')
attention = get_log_from('attention.b1.log')
seq2seq = get_log_from('seq2seq.b1.log')

# sys_branch = ['baseline', 'branch in cuda', 'both branches', 'launch by host', 'best + small op out']
blockdrop = get_log_from('blockdrop.b1.log')
skipnet = get_log_from('skipnet.b1.log')

sys_recursive = ['CocktailerBase', 'serial schedule', 'stack in global memory', 'stack in shared memory', 'parallel schedule']
rae = [
    parse_time(f'../reproduce_results/Figure{figure_id}/base/rae.b1.log'),
    parse_time(f'../reproduce_results/Figure{figure_id}/schedule/rae.opt1.b1.log'),
    parse_time(f'../reproduce_results/Figure{figure_id}/schedule/rae.opt2.b1.log'),
    parse_time(f'../reproduce_results/Figure{figure_id}/schedule/rae.opt3.b1.log'),
    parse_time(f'../reproduce_results/Figure{figure_id}/schedule/rae.opt4.b1.log'),
]

print("lstm", lstm)
print("nasrnn", nasrnn)
print("attention", attention)
print("seq2seq", seq2seq)
print("blockdrop", blockdrop)
print("skipnet", skipnet)
print("rae", rae)

print("loop schedule over base", geo_mean([lstm[0] / lstm[1], nasrnn[0] / nasrnn[1], attention[0] / attention[1], seq2seq[0] / seq2seq[1]]))
print("loop opt speedup", "lstm", lstm[1] / lstm[2], "attention", attention[1] / attention[2])
print("branch schedule over base", "blockdrop", blockdrop[0] / blockdrop[1], "skipnet", skipnet[0] / skipnet[1])
print("branch opt speedup", "blockdrop", blockdrop[1] / blockdrop[2], "skipnet", skipnet[1] / skipnet[2])
print("recursive schedule over base", "rae", rae[0] / rae[1])
print("recursive opt speedup", "rae", "{:.2f}".format(rae[1] / rae[2]), "{:.2f}".format(rae[1] / rae[3]), "{:.2f}".format(rae[3] / rae[4]))

def plot_subfigure(data, ax, model_name, sys):
    # plt.subfigure(1, 6, idx)
    # print(type(ax))
    values = [0 if isinstance(x, str) else x for x in data]
    # text = [x if isinstance(x, str) else '' for x in data]
    # x_id = list(range(6))
    for i in range(len(sys)):
        ax.bar([i * 1.5], [values[i]], label=sys[i], color=color_general[i], hatch=hatch_general[i], edgecolor='black')
    # best_baseline = list(filter(lambda x: isinstance(x, float), values[:-1]))
    # speedup = min(best_baseline) / values[-1]
    # ax.text(len(sys) - 1.5, values[-1], "{:.2f}x".format(speedup))
    for i, dt in enumerate(data):
        if isinstance(dt, str):
            ax.text(i* 1.5, 0.5, dt, rotation='vertical')
    ax.xaxis.set_minor_locator(MultipleLocator(1.5))
    ax.xaxis.set_major_locator(NullLocator())
    ax.set_ylim((0, max(values) * 1.1))
    ax.set_title(model_name, y = -0.4)


def plot_figure(data, idx, model_name, sys):
    # plt.subfigure(1, 6, idx)
    ax = axes
    # print(type(ax))
    values = [0 if isinstance(x, str) else x for x in data]
    # text = [x if isinstance(x, str) else '' for x in data]
    # x_id = list(range(6))
    for i in range(len(sys)):
        ax.bar([i * 1.5], [values[i]], label=sys[i], color=color_rae[i], hatch=hatch_rae[i], edgecolor='black')
    # best_baseline = list(filter(lambda x: isinstance(x, float), values[:-1]))
    # speedup = min(best_baseline) / values[-1]
    # ax.text(len(sys) - 1.5, values[-1], "{:.2f}x".format(speedup))
    # for i, dt in enumerate(data):
    #     if isinstance(dt, str):
    #         ax.text(i, 0.5, dt, rotation='vertical')
    ax.xaxis.set_minor_locator(MultipleLocator(1.5))
    ax.xaxis.set_major_locator(NullLocator())
    ax.set_ylim((0, max(values) * 1.1))
    # ax.set_xlim((-1, len(sys)))
    ax.set_title(model_name, y = -0.3)

# loop figure
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
fig = plt.figure(figsize=(8, 3.5))
axes = fig.subplots(nrows=2, ncols=2) # sharey=True
plt.subplots_adjust(hspace=0.5)
# gs = gridspec.GridSpec(2, 4)

plot_subfigure(lstm, plt.subplot(2, 2, 1), 'LSTM', sys_general)
plot_subfigure(nasrnn, plt.subplot(2, 2, 2), 'NASRNN', sys_general)
plot_subfigure(attention, plt.subplot(2, 2, 3), 'Attention', sys_general)
plot_subfigure(seq2seq, plt.subplot(2, 2, 4), 'Seq2seq', sys_general)

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper center', ncol=len(sys_general), bbox_to_anchor=(0.5, 1.05))
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel("time (ms)")
plt.savefig(os.path.join(plot_dir, f'figure{figure_id}.breakdown_loop.pdf'), bbox_inches='tight')

# branch figure
fig = plt.figure(figsize=(8, 1.5))
axes = fig.subplots(nrows=1, ncols=2) # sharey=True

plot_subfigure(blockdrop, axes[0], 'BlockDrop', sys_general)
plot_subfigure(skipnet, axes[1], 'SkipNet', sys_general)

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper center', ncol=len(sys_general), bbox_to_anchor=(0.5, 1.3))
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel("time (ms)")
plt.savefig(os.path.join(plot_dir, f'figure{figure_id}.breakdown_branch.pdf'), bbox_inches='tight')

# recursion figure
figsize = {
    # "figure.figsize": (16, 2),
    'font.sans-serif': 'Times New Roman',
    'axes.labelsize': 24,
    'font.size': 24,
    'legend.fontsize': 17,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}
plt.rcParams.update(figsize)
fig = plt.figure(figsize=(8, 2))
axes = fig.subplots(nrows=1, ncols=1) # sharey=True

plot_figure(rae, 0, 'RAE', sys_recursive)
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'center right', ncol=1, bbox_to_anchor=(1.32, 0.45))
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel("time (ms)")
plt.savefig(os.path.join(plot_dir, f'figure{figure_id}.breakdown_recursion.pdf'), bbox_inches='tight')
if SHOW: plt.show()
