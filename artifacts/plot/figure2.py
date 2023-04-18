import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator, MaxNLocator
from common import *
import os

figure_id = 2

sys = ['trace', 'with control flow']
seq2seq = [
    parse_time('../reproduce_results/Figure15/jax/seq2seq.unroll.log'),
    parse_time('../reproduce_results/Figure15/jax/seq2seq.fix.log'),
]
blockdrop = [
    parse_time('../reproduce_results/Figure16/jax/blockdrop.unroll.log'),
    parse_time('../reproduce_results/Figure16/jax/blockdrop.fix.log'),
]
rae = [
    parse_time('../reproduce_results/Figure18/jax/rae.unroll.log'),
    parse_time('../reproduce_results/Figure18/jax/rae.fix.log'),
]

colors = [color_def[1], color_def[2]]
hatch_def = ['//', '\\\\']

def plot_subfigure(data, idx, model_name):
    ax = axes[idx]
    for i in range(len(sys)):
        ax.bar([i * 1.5], [data[i]], label=sys[i], color=colors[i], hatch=hatch_def[i],
        edgecolor='black')
    
    ax.xaxis.set_minor_locator(MultipleLocator(1.5))
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlim(-1, 2.5)

    ax.set_title(model_name, y = -0.3)
    if idx == 0:
        ax.set_ylabel("time (ms)")

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
fig = plt.figure(figsize=(8, 1.5))
# fig.constrained_layout()
axes = fig.subplots(nrows=1, ncols=3)
plot_subfigure(seq2seq, 0, '(a) Seq2Seq')
plot_subfigure(blockdrop, 1, '(b) BlockDrop')
plot_subfigure(rae, 2, '(c) RAE')
print("overhead", 1-seq2seq[0]/seq2seq[1], 1-blockdrop[0]/blockdrop[1], 1-rae[0]/rae[1])

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper center', ncol=len(sys), bbox_to_anchor=(0.5, 1.3))
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.savefig(f"{os.path.join(plot_dir, f'figure{figure_id}.overhead_jax.pdf')}", bbox_inches='tight')