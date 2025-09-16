import json
import numpy as np
import matplotlib.pyplot as plt

def plot_radar(ax, title, data, categories, colors):
    """
    Creates a single radar plot on a given matplotlib axis.
    """
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    ax.set_yticklabels([])

    for k_val, values in data.items():
        stats = values.tolist()
        stats += stats[:1]
        ax.plot(angles, stats, color=colors[k_val], linewidth=1.5, label=k_val)
        if k_val in ['k=7', 'k=15']:
            ax.fill(angles, stats, color=colors[k_val], alpha=0.2)

    ax.set_title(title, y=1.1, fontdict={'size': 12})
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))


def main():
    with open('frequencies_dict.json', 'r') as f:
        raw_data = json.load(f)

    processed_data = {"ATTN": {}, "MLP": {}}
    for k_val, samples in raw_data.items():
        attn_freqs = [d['ATTN'] for d in samples]
        mlp_freqs = [d['MLP'] for d in samples]
        processed_data["ATTN"][k_val] = np.mean(attn_freqs, axis=0)
        processed_data["MLP"][k_val] = np.mean(mlp_freqs, axis=0)

    num_layers = len(processed_data["ATTN"]["k=3"])
    layer_labels = [str(i) for i in range(num_layers)]
    plot_colors = {'k=3': 'red', 'k=7': 'cyan', 'k=15': 'green'}

    fig, axes = plt.subplots(figsize=(18, 8), nrows=1, ncols=2, subplot_kw=dict(projection='polar'))

    plot_radar(axes[0], 'Attention (Positive set)', processed_data['ATTN'], layer_labels, plot_colors)
    plot_radar(axes[1], 'MLP (Positive set)', processed_data['MLP'], layer_labels, plot_colors)

    fig.suptitle('Top-k Layer Frequencies by Component (k={3,7,15}) — Layers as Edges', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('freq_radar_plot.pdf')
    plt.show()

if __name__ == '__main__':
    main()