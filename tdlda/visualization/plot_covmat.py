import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from math import ceil


def _symlog(arr):
    arr = np.where((arr > 1), np.log(arr) + 1, arr)
    arr = np.where((arr < -1), -np.log(-arr) - 1, arr)
    return arr


def plot_covmat(covmat, n_channels, n_ivals, title='Covariance matrix', unit=None, add_zoom=False, show_colorbar=False,
                channel_names=None, scaling=(2, 98), axes=None):
    # scaling can be 'symlog' or a tuple which defines the percentiles
    cp = sns.color_palette()
    assert covmat.shape[0] == n_channels * n_ivals,\
        "Covariance does not correspond to feature dimensions.\n" \
        f"Cov-dim: {covmat.shape} vs. n_channels: {n_channels} and n_ivals: {n_ivals}"
    if scaling == 'symlog':
        data_range = (np.min(covmat), np.max(covmat))
        covmat = _symlog(covmat)
        color_lim = np.max(np.abs(covmat))
    elif scaling is None:
        data_range = (np.min(covmat), np.max(covmat))
        color_lim = np.max(np.abs(data_range))
    else:
        data_range = (np.min(covmat), np.max(covmat))
        percs = np.percentile(covmat, scaling)
        color_lim = np.max(np.abs(percs))
    base_fig_width, base_fig_height = 4, 3
    if add_zoom:
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(2*base_fig_width, base_fig_height))
        else:
            fig = axes[0].figure
            ax = axes[0]
    else:
        if axes is None:
            fig, ax = plt.subplots(1, 1, figsize=(base_fig_width, base_fig_height))
        else:
            fig = axes.figure
            ax = axes
    im_map = ax.imshow(covmat, cmap='RdBu_r', vmin=-color_lim, vmax=color_lim)
    if title is not None:
        title_str = title + '\n'
        title_str += f'Original datarange: {data_range}\n'
        if scaling != 'symlog' and scaling is not None:
            title_str += f'Percentiles: {percs}'
        ax.set_title(title_str)
    ax.grid(False)
    xticks = [i*n_channels + ceil(n_channels / 2) for i in range(n_ivals)]
    xtick_labels = [f'$T_{{{i+1}}}$' for i in range(n_ivals)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks(xticks)
    ax.set_yticklabels(xtick_labels)
    for i in range(n_ivals):
        x1 = i*n_channels - 0.5
        x2 = x1+n_channels
        y1 = x1
        y2 = x2
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], clip_on=True, color=cp[2], linewidth=1)
    if show_colorbar:
        fc = fig.colorbar(im_map)
        if unit is not None:
            fc.ax.set_title(unit, rotation=0)
    if add_zoom:
        covmat_zoom = covmat[0:n_channels, 0:n_channels]
        ax = axes[1]
        im_map = ax.imshow(covmat_zoom, cmap='RdBu_r', vmin=-color_lim, vmax=color_lim)
        ax.set_title(f'Zoom on first main diagonal block $B_1$')
        ax.grid(False)
        xticks = list(range(n_channels))
        if channel_names is None:
            xtick_labels = [f'ch_{i}' for i in range(n_channels)]
        else:
            if len(channel_names) != n_channels:
                raise ValueError('Number of channel names do not correspond to the number of channels.')
            xtick_labels = channel_names
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=2.5)
        ax.set_yticks(xticks)
        ax.set_yticklabels(xtick_labels, fontsize=2.5)
        [s.set_color(cp[2]) for s in ax.spines.values()]
        if show_colorbar:
            fig.colorbar(im_map)
    # fig.tight_layout()