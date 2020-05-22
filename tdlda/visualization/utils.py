from matplotlib import pyplot as plt


def add_color_spans(ax, color_spans, color='grey'):
    for i, cs in enumerate(color_spans):
        even = (i + 1) % 2 == 0
        ax.axvspan(cs[0], cs[1], alpha=0.15 if even else 0.3, facecolor=color, edgecolor=None)


def add_baseline(ax, baseline, y_extent=20):
    rect = plt.Rectangle((baseline[0], -y_extent), baseline[1] - baseline[0], 2*y_extent,
                         facecolor='grey', edgecolor=None, zorder=-20, alpha=0.25)
    ax.add_patch(rect)