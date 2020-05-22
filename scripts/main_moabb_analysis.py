import pandas as pd
import yaml
import os
from pathlib import Path
import numpy as np
import argparse
import seaborn as sns
import shutil
from tdlda.benchmark.visualization import _ds_pretty, _jm_pretty, _jm_and_pipe_pretty, plot_benchmark_results
import re
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib import pyplot as plt

sns.set_palette("colorblind")
sns.set(style="whitegrid")

def best_pipeline_of_family_for_dataset(ds_pd, family_identifier, error_action='warn'):
    fam_pd = ds_pd[ds_pd['pipeline'].str.contains(re.escape(family_identifier))]
    nan_thresh = 50
    for pipeline in fam_pd['pipeline'].unique():
        cur_pipe = fam_pd.loc[fam_pd['pipeline'] == pipeline]
        pipe_nan_perc = 100 * cur_pipe['score'].isna().sum() / len(cur_pipe)
        if pipe_nan_perc >= nan_thresh or np.isnan(pipe_nan_perc):
            print(f'{pipeline} has {pipe_nan_perc:1.2f}% nans, which exceeds nan_threshold of {nan_thresh}%.'
                  f' Removing from analysis.')
            fam_pd = fam_pd.loc[~(fam_pd['pipeline'] == pipeline)]
    fam_pd = fam_pd.groupby('pipeline').mean()
    fam_pd = fam_pd.sort_values(by='score', ascending=False)
    if not fam_pd.empty:
        best_pipeline_name = fam_pd.iloc[0:1].index[0]
        best_pipeline_pd = ds_pd[ds_pd['pipeline'].str.contains(best_pipeline_name, regex=False)]
        best_pipeline_pd = best_pipeline_pd.replace(best_pipeline_name, f'zref_{best_pipeline_name}')
        return best_pipeline_pd
    else:
        fail_str = f'Did not find a dataset using family identifier: {family_identifier}'
        if error_action == 'warn':
            print(f'Warning: {fail_str}')
        elif error_action == 'raise':
            raise ValueError(fail_str)
        else:
            raise ValueError('Unknown error_action.')
        return None


plt.rcParams['backend'] = 'QT4Agg'
plt.ioff()

LOCAL_CONFIG_FILE = f'local_config.yaml'

try:
    with open(LOCAL_CONFIG_FILE, 'r') as conf_f:
        local_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)
except OSError:
    os.chdir('scripts')
    with open(LOCAL_CONFIG_FILE, 'r') as conf_f:
        local_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)

RESULTS_ROOT = Path(local_cfg['results_root']) / local_cfg['benchmark_meta_name']

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('results_path',
                    help=f'Grouping and name of the results, e.g.: 2020-03-02/e88707f1-caa7-4318-92d0-3cd6acef8c2f')
args = parser.parse_args()

RESULTS_FOLDER = RESULTS_ROOT / args.results_path
PLOTS_FOLDER = RESULTS_FOLDER / 'plots'

ANALYSIS_CONFIG_FILE = f'analysis_config.yaml'

with open(RESULTS_FOLDER / ANALYSIS_CONFIG_FILE, 'r') as conf_f:
    ana_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)  # todo use this somewhere

os.makedirs(PLOTS_FOLDER, exist_ok=True)

sub_results = []

for csv_f in RESULTS_FOLDER.glob('*.csv'):
    sub_results.append(pd.read_csv(csv_f))

r = pd.concat(sub_results, ignore_index=True)

r['dataset'] = r['dataset'].replace({
    '003-2015': 'BNCI_healthy_2015',
    'Spot Pilot P300 dataset single': 'Spot_single_trial',
    '008-2014': 'BNCI_ALS_patients',
    'EPFL P300 dataset': 'EPFL',
    '009-2014': 'BNCI_healthy_2014',
    'Spot Pilot P300 dataset pooled': 'Spot_pooled',
    'Brain Invaders 2013a': 'Brain_invaders',
    'Aphasia Direction Study dataset words': 'Aphasia_direction_words',
    'Aphasia Direction Study dataset oddball': 'Aphasia_direction_oddball',
    'Aphasia Main Study dataset oddball': 'Aphasia_main_oddball',
    'Aphasia Main Study dataset words bci': 'Aphasia_main_words_bci',
    'Aphasia Main Study dataset words button': 'Aphasia_main_words_button',
    'Visual Speller LLP': 'Visual_LLP',
    'Visual Speller MIX': 'Visual_MIX',
})

print(f'Analysis done for {len(r["dataset"].unique())} datasets.')
print(r["dataset"].unique())

#%% data manipulation and cleanup

pipelines = r['pipeline'].unique()
threshold_nans_percent = 10
for p in pipelines:
    asd = r.loc[r['pipeline'] == p]
    nan_perc = 100*asd['score'].isna().sum() / len(asd)
    print(f'{p}: {nan_perc:1.2f}% NaNs')
    if nan_perc >= threshold_nans_percent:
        print(f'{p} exceeds nan_threshold of {threshold_nans_percent}. Removing from analysis.')
        r = r.loc[~(r['pipeline'] == p)]
r = r.sort_values(by=['pipeline'])

highscores = r.groupby(['pipeline', 'dataset', 'subject']).mean() \
    .groupby(['pipeline', 'dataset']).mean().groupby(['pipeline']).mean().sort_values(by='score')
# highscores = r.groupby('pipeline').mean().sort_values(by='score')
print(highscores)
highscores.to_csv(PLOTS_FOLDER / 'highscores_all_datasets.csv', index=True, float_format='%.4f')

# %% Highscores
bounds = [('tiny', 0, 100), ('small', 100, 250), ('medium', 250, 700), ('large', 700, np.inf)]

for b in bounds:
    print(b)
    r_sub = r[(r['samples'] >= b[1]) & (r['samples'] < b[2])]
    highscores = r_sub.groupby(['pipeline', 'dataset', 'subject']).mean()\
        .groupby('pipeline').mean().sort_values(by='score')
    highscores.to_csv(PLOTS_FOLDER / f'highscores_{b[0]}_datasets.csv', index=True, float_format='%.4f')

#%%
# Split EPFL dataset into two datasets: healthy vs. patients
r.loc[np.logical_and(r['dataset'] == 'EPFL', r['subject'] <= 4), 'dataset'] = 'EPFL_disabled'
r.loc[np.logical_and(r['dataset'] == 'EPFL', r['subject'] > 4), 'dataset'] = 'EPFL_healthy'

# Split Brain invaders into two datasets: single-session and multi-session
r.loc[np.logical_and(r['dataset'] == 'Brain_invaders', r['subject'] <= 7), 'dataset'] = 'Brain_invaders_multisession'
r.loc[np.logical_and(r['dataset'] == 'Brain_invaders', r['subject'] > 7), 'dataset'] = 'Brain_invaders_singlesession'
# %%
r_bkp = r
for ds in r['dataset'].unique():
    print(f'\n\n\n======={ds}=======')
    sub = r.loc[r['dataset'] == ds]
    sub = sub[sub['pipeline'].str.contains('_rg_')]
    temp = sub.groupby('pipeline').mean()
    temp = temp.sort_values(by='score', ascending=False)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 180):
        print(temp.iloc[0:10])

# datasets = ['003-2015', 'Spot Pilot P300 dataset', '008-2014', 'EPFL P300 dataset', '009-2014',
#             'Spot Pilot P300 dataset pooled', 'Brain Invaders 2013a']
# dataset_labels = ['BNCI_Healthy_2015', 'Spot single trial']

dims = ['jm_few', 'jm_standard', 'jm_numerous', 'sel_all']
# dim_label = ['$T_2$', '$T_5$', '$T_{10}$', '$T_{all}$']

for ds in r['dataset'].unique():
    sub = r.loc[r['dataset'] == ds]
    ref_pipeline_family = ['_rg_', '_kPCA']
    ref_pipeline_family = ['ceat_rg_xdawncomps_5_xdawnclasses_Target', 'jm_numerous_kPCA(70)_skl_lsqr']
    ref_pipes = []
    for rpf in ref_pipeline_family:
        ref_pipes.append(best_pipeline_of_family_for_dataset(sub, rpf, error_action='warn'))

    for d_i, d_str in enumerate(dims):
        d_pd = sub[sub['pipeline'].str.contains(f'{d_str}')]
        d_pd = d_pd[~d_pd['pipeline'].str.contains(f'_kPCA')]
        for rp in ref_pipes:
            d_pd = pd.concat([d_pd, rp]) if rp is not None else d_pd

        # # COMPLETE VERSION OF PLOT FUNCTION CALL
        # plot_benchmark_results(d_pd, dataset=ds, jm_dim=d_str, save=True, dim_prefix=d_str, ylim='auto',
        #                        output_dir=PLOTS_FOLDER, session_mean=True, out_prefix='')
        # # PAPER VERSION OF PLOT FUNCTION CALL
        # paper_pipes = ['changeblock_standard_cov_no_chshr', 'skl_lsqr', 'default_bbci', '_rg_', '_kPCA']
        # d_pd = d_pd[d_pd['pipeline'].str.contains('|'.join(paper_pipes))]
        plot_benchmark_results(d_pd, dataset=ds, jm_dim=d_str, save=True, dim_prefix=d_str, ylim='auto',
                               output_dir=PLOTS_FOLDER, session_mean=True, figsize=(4, 3), out_prefix='')
    plt.close("all")


# %%
REF_PIPELINES = [
    'jm_numerous_lda_p_cov',
    'ceat_rg_xdawncomps_5_xdawnclasses_Target',
    ]
force_reload = False
r_temp = r
for REF_PIPELINE in REF_PIPELINES:

    ref_path = PLOTS_FOLDER / f'REF_{REF_PIPELINE}.csv.gz'
    if not ref_path.exists() or force_reload:
        dataset_subject = r_temp[['dataset', 'subject', 'session']].drop_duplicates().reset_index(drop=True)
        subsets = []
        for row in dataset_subject.itertuples():
            print(f'{row.Index} of {len(dataset_subject)} ({100*row.Index / len(dataset_subject):1.2f} %)')
            subset = r_temp.loc[np.logical_and(r_temp['dataset'] == row.dataset, np.logical_and(
                                r_temp['subject'] == row.subject, r_temp['session'] == row.session))]
            ref_score = float(subset.loc[subset['pipeline'] == REF_PIPELINE]['score'])
            subset.loc[:, 'score'] -= ref_score
            subsets.append(subset)

        r_ref = pd.concat(subsets)
        r_ref.to_csv(ref_path, index=False)
    else:
        r_ref = pd.read_csv(ref_path)
    # %%
    pool_over_subjects = False
    temp = r_ref
    pipelines = list(temp['pipeline'].unique())
    ACROSS_DATASET_PLOTS_FOLDER = PLOTS_FOLDER / 'across_datasets'
    SELECTED_FOLDER = PLOTS_FOLDER / 'across_datasets' / 'selected_pipelines'
    os.makedirs(ACROSS_DATASET_PLOTS_FOLDER, exist_ok=True)
    os.makedirs(SELECTED_FOLDER, exist_ok=True)
    n_channel_configurations = len(temp['channels'].unique())

    with sns.color_palette(sns.light_palette("navy", n_colors=6)[1:]):
        for compare_pipeline in pipelines:
            plt.close("all")
            fig, ax = plt.subplots(1, 1, figsize=(9, 4))
            asd_pool = temp.groupby(['dataset', 'pipeline']).aggregate([np.mean, np.std]).reset_index()
            asd_pool = asd_pool.loc[asd_pool['pipeline'] == compare_pipeline]
            col_order = asd_pool.sort_values(by=('samples', 'mean'))['dataset']
            asd_single = temp.groupby(['dataset', 'subject', 'pipeline']).aggregate([np.mean, np.std]).reset_index()
            asd_single = asd_single.loc[asd_single['pipeline'] == compare_pipeline]
            asd = asd_pool if pool_over_subjects else asd_single
            scatter_alpha = 1 if pool_over_subjects else 0.4

            for d in asd['dataset'].unique():
                replace_dict = {
                    d: f'{_ds_pretty(d, bold=True)} \n({asd_pool.loc[asd_pool["dataset"] == d]["samples"]["mean"].iloc[0]:1.0f}'
                       f'$\\pm${asd_pool.loc[asd_pool["dataset"] == d]["samples"]["std"].iloc[0]:1.1f})'
                }
                asd = asd.replace(replace_dict)
                col_order = col_order.replace(replace_dict)
            ax.axhline(0, color='k', linestyle='--', alpha=0.8)

            ax.set_title(f'{_jm_and_pipe_pretty(compare_pipeline)[1]} $\\bf{{vs.}}$ {_jm_and_pipe_pretty(REF_PIPELINE)[1]}')
            fig.tight_layout()
            sns.swarmplot(data=asd, x='dataset', y=('score', 'mean'), ax=ax, alpha=1, hue=('channels', 'mean'),
                          order=col_order)
            ax.set_ylabel(f'Mean AUC difference')
            ax.set_xlabel(f'Dataset')
            ax.xaxis.grid(True)
            for tick in ax.get_xticklabels():
                tick.set_ha('right')
                tick.set_rotation_mode('anchor')
                tick.set_rotation(45)
            plt.legend(title='Channels', ncol=n_channel_configurations, loc='best')

            fig.subplots_adjust(bottom=0.4, left=0.1)
            fname = f'REF_{REF_PIPELINE}_VS_{compare_pipeline}.png'
            fname_pdf = f'REF_{REF_PIPELINE}_VS_{compare_pipeline}.pdf'
            fig.savefig(ACROSS_DATASET_PLOTS_FOLDER / fname, dpi=200)
            # ensure that a simple rerun without changes in the plot, does not trigger git changes (for paper)
            with PdfPages(ACROSS_DATASET_PLOTS_FOLDER / fname_pdf, metadata={'CreationDate': None}) as pdf:
                pdf.savefig(fig)
