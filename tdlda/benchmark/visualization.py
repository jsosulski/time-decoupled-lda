from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import re
import random
from matplotlib.lines import Line2D

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

# from matplotlib import rc
# # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
# rc('text', usetex=True)

def _ds_pretty(ds_name, ignore_on_error=True, bold=False):
    dataset_title_mapping = {
        'BNCI_healthy_2015': 'BNCI\\_healthy\\_2',
        'Spot_single_trial': 'SPOT',
        'BNCI_ALS_patients': 'BNCI\\_patient',
        'EPFL_healthy': 'EPFL\\_healthy*',
        'EPFL_disabled': 'EPFL\\_patient*',
        'BNCI_healthy_2014': 'BNCI\\_healthy\\_1',
        'Brain_invaders_multisession': 'BI\\_a^+',
        'Brain_invaders_singlesession': 'BI\\_b^+',
        'Aphasia_direction_words': 'WORD\\_healthy',
        'Aphasia_direction_oddball': 'TONE\\_healthy',
        'Aphasia_main_oddball': 'TONE\\_patient',
        'Aphasia_main_words_bci': 'WORD\\_patient',
        # 'Aphasia_main_words_button': 'WORD_patient',
        'Visual_LLP': 'SPELLER\\_LLP',
        'Visual_MIX': 'SPELLER\\_MIX',
    }
    if ds_name in dataset_title_mapping:
        return f'$\\bf{{{dataset_title_mapping[ds_name]}}}$'
    elif ignore_on_error:
        return ds_name
    else:
        raise ValueError(f'{ds_name} is unknown and cannot be mapped to pretty name.')


def _jm_pretty(jm_name):
    jm_title_mapping = {
        'jm_few': '$T_2$',
        'jm_standard': '$T_5$',
        'jm_numerous': '$T_{10}$',
        'sel_all': '$T_{all}$',
        'ceat_rg': '',
    }
    return f'{jm_title_mapping[jm_name]}'


def _pipe_pretty(pipe_name, with_hypers=False):
    jm_title_mapping = {
        'lda_imp_p_cov': 'LDA imp. p-cov',
        'lda_p_cov': 'LDA p-cov',
        'lda_c_covs': 'LDA c-covs',
    }
    if pipe_name in jm_title_mapping:
        return f'{jm_title_mapping[pipe_name]}'
    elif 'xdawncomps' in pipe_name or pipe_name in ['lda_henrich', 'lda_default', 'lr_henrich']:
        if with_hypers:
            m = re.match('xdawncomps_(\d)_xdawnclasses_(.*)', pipe_name)
            if m is None:
                return pipe_name
            xdawn_components = m.group(1)
            xdawn_classes = 'Target' if m.group(2) == 'Target' else 'Both'
            return f'Riemann$_{{{xdawn_components}, {xdawn_classes}}}$'
        else:
            return 'Riemann'
    elif 'kPCA' in pipe_name:
        if with_hypers:
            for substr in pipe_name.split('_'):
                if len(substr) > 4 and substr[0:4] == 'kPCA':
                    return substr.replace('None', 'all').replace('(', '$_{').replace(')', '}$')  # LDA c-covs')
        else:
            return 'kPCA'

    else:
        # raise ValueError(f'Unknown pipe name: {pipe_name}.')
        return pipe_name


def _split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])


def _jm_and_pipe_pretty(text):
    if text[0:4] == 'zref':
        text = text[5:]
    jm, pipe = _split_at(text, '_', 2)
    return _jm_pretty(jm), _pipe_pretty(pipe)


def plot_benchmark_results(results, dataset=None, dim_prefix='', jm_dim=None, ylim=None, save=False, output_dir=None,
                           out_prefix=None, session_mean=False, figsize=(9, 6), plot_legend=False):
    results = results.replace(f'{dim_prefix}_', '')
    for p in results['pipeline'].unique():
        jm_pretty, pipe_pretty = _jm_and_pipe_pretty(p)
        if pipe_pretty[0:4] == 'kPCA':
            pipe_pretty = 'kPCA'
        replace_dict = {
            p: pipe_pretty
        }
        results = results.replace(replace_dict)
    pipes = results['pipeline'].unique()
    custom_order = ['LDA imp. p-cov', 'LDA p-cov', 'LDA c-covs', pipes[-2], pipes[-1]]
    custom_order = [c for c in custom_order if c in pipes]
    if 'Riemann' == custom_order[-1]:
        custom_order[-2], custom_order[-1] = custom_order[-1], custom_order[-2]
    if session_mean:
        results = results.groupby(['subject', 'pipeline']).aggregate([np.mean]).reset_index()
        results.columns = [x[0] for x in results.columns]
    ax, leg_handles = plot_matched_wrapper(data=results, x='pipeline', y='score', x_order=custom_order,
                                          match_col='subject', x_match_sort='LDA imp. p-cov', figsize=figsize)
    if plot_legend:
        leg_labels = [str(i) for i in range(1, len(leg_handles) + 1)]
        leg_ncols = len(leg_labels) // 10 + 1
        plt.legend(handles=leg_handles, labels=leg_labels, title='Subj.', ncol=leg_ncols, prop={'size': 6},
                   loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('Mean AUC')
    ax.set_xlabel('Classification method')
    save_filename = f'{dataset}__{jm_dim}'
    title = f'{_ds_pretty(dataset)}'# using {_jm_pretty(jm_dim)}'
    ax.set_title(title)
    fig = ax.figure
    fig.tight_layout()
    if type(ylim) is tuple and len(ylim) == 2:
        ax.set_ylim(ylim[0], ylim[1])
    elif ylim == 'auto':
        ax.set_ylim(auto=True)
    if save and output_dir is not None:
        fig.savefig(output_dir / f'{out_prefix}{save_filename}.png', dpi=200)
        with PdfPages(output_dir / f'{out_prefix}{save_filename}.pdf', metadata={'CreationDate': None}) as pdf:
            pdf.savefig(fig)


def plot_matched_wrapper(data=None, x=None, y=None, x_order=None, match_col=None, x_match_sort=None, title=None,
                         ax=None, figsize=(9, 6)):
    if ax is None:
        fig, ax = plt.subplots(1, 1, facecolor='white', figsize=figsize, )
    if match_col is not None:
        ax, leg_handles = plot_matched(data=data, x=x, y=y, x_order=x_order, match_col=match_col,
                                       x_match_sort=x_match_sort, title=title, ax=ax, figsize=figsize, sort_marker='')
    else:
        ax = sns.stripplot(data=data, y=y, x=x, ax=ax, jitter=True, alpha=0.2, zorder=1,
                           order=x_order)
        leg_handles = None
    return ax, leg_handles


def plot_matched(data=None, x=None, y=None, x_order=None, match_col=None, x_match_sort=None, title=None, ax=None,
                 figsize=(9, 6), sort_marker='●', error='amend'):
    if ax is None:
        fig, ax = plt.subplots(1, 1, facecolor='white', figsize=figsize)
    if x_order is not None:
        data = data.loc[data[x].str.match('|'.join([re.escape(xo) for xo in x_order]))]
        ux = data[x].unique()
        clean_order = [x_order[i] for i in range(len(x_order)) if x_order[i] in ux]
        if len(clean_order) < len(x_order):
            print('Warning: Truncating ordering, as some pipelines not existing in data frame.')
            x_order = clean_order
    if x_match_sort not in x_order:
        errorwarn_string = f'Cannot sort by {x_match_sort} as it is not in {x_order} / data array.'
        if error == 'raise':
            raise ValueError(errorwarn_string)
        else:
            print(f'WARNING: {errorwarn_string}')
            x_match_sort = None
    cp = sns.color_palette()
    marker_arr = ['^', 's', 'p', 'H', 'o']
    n_markers = len(marker_arr)
    num_x = len(data[x].unique())
    num_matched = len(data[match_col].unique())
    sort_idx = None
    if x_match_sort is not None:
        sort_idx = data.loc[data[x] == x_match_sort].sort_values(by=match_col, ascending=True).reset_index().\
            sort_values(by='score', ascending=True).index.copy()
    c_offs_left = 2
    c_offs_right = 2
    gmap = LinearSegmentedColormap.from_list('custom', [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)],
                                             N=(num_matched // n_markers + 1 + c_offs_left + c_offs_right))
    legend_markers = []
    for i, x_main in enumerate(x_order):
        base_col = cp[i]
        cmap = LinearSegmentedColormap.from_list('custom', [(0, 0, 0), base_col, (1, 1, 1)],
                                                 N=(num_matched // n_markers + 1 + c_offs_left + c_offs_right))
        r = data.loc[data[x] == x_main]  # .sort_values(by=match_col, ascending=True).reset_index()
        # if sort_idx is None:
        #     sort_idx = r.sort_values(by='score', ascending=True).index.copy()
        if sort_idx is not None:
            r = r.iloc[sort_idx]
        x_center = i + 1
        x_width = 0.45
        x_space = np.linspace(x_center - x_width, x_center + x_width, num_matched)
        m_score, m_std = r.aggregate((np.mean, np.std))[y]
        m_err = m_std / np.sqrt(num_matched)
        err_artists = ax.errorbar(x_center, m_score, 2 * m_err, capsize=0, zorder=15, color='k', linewidth=1, alpha=1,
                                  dash_capstyle='round')
        err_artists[2][0].set_capstyle('round')
        ax.scatter(x_center, m_score, marker='X', color=base_col, edgecolor='k', linewidth=0.4, zorder=20, s=20)
        for j, x_j in enumerate(x_space):
            score = r[y].iloc[j]
            m = marker_arr[j % len(marker_arr)]
            sdef = 35
            s = sdef * 0.66 if m in ['s', 'D'] else sdef  # these two markers are unexplicably larger in default pyplot
            ax.scatter(x_j, score, alpha=.8, marker=m, linewidth=0.4, edgecolor=(0.8, 0.8, 0.8), s=s,
                       color=cmap((j // n_markers) + c_offs_left))
            if i == 0:
                legend_markers.append(Line2D([0], [0], marker=m, color='w', label=r[match_col].iloc[j],
                                      markerfacecolor=gmap((j // n_markers) + c_offs_left), markersize=1.2*np.sqrt(s)))
                # need to translate markersize between scatter and plt function, 1.2*sqrt() seems to work kind of
    ax.set_xticks(np.arange(1, num_x + 1))
    if x_match_sort is not None:
        x_order[x_order.index(x_match_sort)] = sort_marker + x_order[x_order.index(x_match_sort)]
    ax.set_xticklabels(x_order)
    ax.set_xlim(0, num_x + 1)
    for i, tick in enumerate(ax.get_xticklabels()):
        if len(tick.get_text()) > 0:
            tick.set_ha('right')
            tick.set_rotation_mode('anchor')
            tick.set_rotation(20)
            tick.set_color(cp[i])
    if title is not None:
        ax.set_title(title)
    return ax, legend_markers
