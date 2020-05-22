import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
from tdlda.benchmark.visualization import _jm_and_pipe_pretty, plot_matched, _ds_pretty
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import yaml
import argparse
import os

def nice_name(text):
    jm, pipe = _jm_and_pipe_pretty(text)
    # return f'{jm}, {pipe}' if len(jm) > 0 else pipe
    return pipe

sns.set(style="whitegrid")
sns.set_palette("colorblind")

LOCAL_CONFIG_FILE = f'local_config.yaml'
with open(LOCAL_CONFIG_FILE, 'r') as conf_f:
    local_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)

RESULTS_ROOT = Path(local_cfg['results_root']) / local_cfg['benchmark_meta_name']

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('results_path', help=f'Grouping and name of the results, e.g.: 2020-03-02/no_art_no_base')
args = parser.parse_args()

RESULTS_FOLDER = RESULTS_ROOT / args.results_path
PLOTS_FOLDER = RESULTS_FOLDER / 'plots'

os.makedirs(PLOTS_FOLDER, exist_ok=True)

SELECTED_PIPELINES = [
    'jm_numerous_lda_imp_p_cov',
    'jm_numerous_lda_p_cov',
    'jm_numerous_lda_c_covs',
    'ceat_rg_xdawncomps_5_xdawnclasses_Target',
    'jm_numerous_kPCA(70)_skl_lsqr',
]

results = []

for csv_f in RESULTS_FOLDER.glob('*.csv'):
    results.append(pd.read_csv(csv_f))
raw_r = pd.concat(results)
raw_r = raw_r[raw_r['pipeline'].isin(SELECTED_PIPELINES)]
raw_r = raw_r.groupby(['pipeline', 'dataset', 'subject']).mean().reset_index()

r = raw_r


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

#%%
# Split EPFL dataset into two datasets: healthy vs. patients
r.loc[np.logical_and(r['dataset'] == 'EPFL', r['subject'] <= 4), 'dataset'] = 'EPFL_disabled'
r.loc[np.logical_and(r['dataset'] == 'EPFL', r['subject'] > 4), 'dataset'] = 'EPFL_healthy'

# Split Brain invaders into two datasets: single-session and multi-session
r.loc[np.logical_and(r['dataset'] == 'Brain_invaders', r['subject'] <= 7), 'dataset'] = 'Brain_invaders_multisession'
r.loc[np.logical_and(r['dataset'] == 'Brain_invaders', r['subject'] > 7), 'dataset'] = 'Brain_invaders_singlesession'

for ds in r['dataset'].unique():
    r = r.replace({
        ds: _ds_pretty(ds)
    })
# # %%
# cp = sns.color_palette()
#

# %%
results = r
cp = sns.color_palette()
fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
for p in results['pipeline'].unique():
    results = results.replace({
        p: nice_name(p)
    })

NICE_PIPELINES = [nice_name(p) for p in SELECTED_PIPELINES]
plot_me = results.groupby(['pipeline', 'dataset']).mean().reset_index()
ax, leg_handles = plot_matched(data=plot_me, ax=ax, x='pipeline', y='score', match_col='dataset',
                               x_order=NICE_PIPELINES, x_match_sort='LDA imp. p-cov', sort_marker='')
ax.set_ylabel('Mean AUC')
ax.set_xlabel('Classification method')

ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.grid(b=True, which='major', linewidth=1.0)
ax.grid(b=True, which='minor', linewidth=0.3)

for tick in ax.get_xticklabels():
    tick.set_rotation_mode("anchor")
    tick.set_rotation(20)
    tick.set_ha('right')

plt.legend(handles=leg_handles, title='Results for all datasets', ncol=3, prop={'size': 6},
           bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0)
fig.tight_layout()
with PdfPages(PLOTS_FOLDER / f'results_across_all_datasets.pdf', metadata={'CreationDate': None}) as pdf:
    pdf.savefig(fig)
plt.show(block=False)
