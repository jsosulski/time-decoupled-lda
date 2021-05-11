from moabb.evaluations import WithinSessionEvaluation
import moabb
import numpy as np
import warnings
from datetime import datetime as dt
import yaml
import argparse
import uuid
import os
from pathlib import Path
from shutil import copyfile
from tdlda.benchmark.utils import create_lda_pipelines, create_ceat_rg_lr_pipelines, create_only_best_paper_pipelines

from tdlda.benchmark.utils import get_benchmark_config
import time

LOCAL_CONFIG_FILE = f'local_config.yaml'
ANALYSIS_CONFIG_FILE = f'analysis_config.yaml'

t0 = time.time()

with open(LOCAL_CONFIG_FILE, 'r') as conf_f:
    local_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)

RESULTS_ROOT = Path(local_cfg['results_root'])
RESULTS_RUN_NAME = os.getenv('RESULTS_RUN_NAME')
RESULTS_GROUPING = os.getenv('RESULTS_GROUPING')
if not RESULTS_RUN_NAME:
    raise ValueError('Expected to have RESULTS_RUN_NAME environment variable set')
if not RESULTS_GROUPING:
    raise ValueError('Expected to have RESULTS_GROUPING environment variable set')
RESULTS_FOLDER = RESULTS_ROOT / local_cfg['benchmark_meta_name'] / RESULTS_RUN_NAME / RESULTS_GROUPING

with open(RESULTS_FOLDER / ANALYSIS_CONFIG_FILE, 'r') as conf_f:  # use copy of the analysis config file
    ana_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)

VALID_DATASETS = ['spot_single', 'epfl', 'bnci_1', 'bnci_als', 'bnci_2', 'braininvaders']

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('dataset', help=f'Name of the dataset. Valid names: {VALID_DATASETS}')
parser.add_argument('subjects_sessions', help='[Optional] Indices of subjects to benchmark.', type=str, nargs='*')
args = parser.parse_args()

print(args)

dataset_name = args.dataset
subject_session_args = args.subjects_sessions
if ' ' in dataset_name and len(subject_session_args) == 0:
    subject_session_args = dataset_name.split(' ')[1:]
    dataset_name = dataset_name.split(' ')[0]
if dataset_name not in VALID_DATASETS:
    raise ValueError(f'Invalid dataset name: {dataset_name}. Try one from {VALID_DATASETS}.')
if len(subject_session_args) == 0:
    subjects = None
    sessions = None
else:  # check whether args have format [subject, subject, ...] or [subject:session, subject:session, ...]
    if np.all([':' in s for s in subject_session_args]):
        subjects = [int(s.split(':')[0]) for s in subject_session_args]
        sessions = [int(s.split(':')[1]) for s in subject_session_args]
    elif not np.any([':' in s for s in subject_session_args]):
        subjects = [int(s.split(':')[0]) for s in subject_session_args]
        sessions = None
    else:
        raise ValueError('Currently, mixed subject:session and only subject syntax is not supported.')
print(f'Subjects: {subjects}')
print(f'Sessions: {sessions}')

start_timestamp_as_str = dt.now().replace(microsecond=0).isoformat().replace(":", "-")

warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

moabb.set_log_level('warn')

np.random.seed(42)

##############################################################################
# Create pipelines
##############################################################################

labels_dict = {'Target': 1, 'NonTarget': 0}

prepro_cfg = ana_cfg['default']['data_preprocessing']

bench_cfg = get_benchmark_config(dataset_name, prepro_cfg, subjects=subjects, sessions=sessions)

if hasattr(bench_cfg['dataset'], 'stimulus_modality'):
    feature_preprocessing_key = bench_cfg['dataset'].stimulus_modality
else:
    feature_preprocessing_key = ana_cfg['default']['fallback_modality']

pipelines = dict()

if local_cfg['benchmark_extent'] == 'full':
    # initialize Riemannian geometry pipelines
    pipelines.update(create_ceat_rg_lr_pipelines(labels_dict))
    # initialize LDA variants
    pipelines.update(create_lda_pipelines(ana_cfg, feature_preprocessing_key, bench_cfg['N_channels']))
elif local_cfg['benchmark_extent'] == 'best_only':
    pipelines.update(create_only_best_paper_pipelines(ana_cfg, feature_preprocessing_key, bench_cfg['N_channels'],
                     labels_dict))
else:
    raise ValueError(f'Invalid benchmark extent.')

# IF YOU WANT TO DOWNLOAD ALL DATA FIRST, UNCOMMENT THE NEXT LINE
#pipelines = dict(test=pipelines['jm_few_lda_p_cov'])
##############################################################################
# Evaluation
##############################################################################

identifier = f'{dataset_name}_subj_{subjects if subjects is not None else "all"}' \
             f'_sess_{sessions if sessions is not None else "all"}_{start_timestamp_as_str}'.replace(' ', '')
unique_suffix = f'{identifier}_{uuid.uuid4()}'
debug_path = RESULTS_FOLDER / f'{identifier}_DEBUG.txt'

with open(debug_path, 'w') as debug_f:
    debug_f.writelines([f'{l}: {os.environ[l]}\n' for l in sorted(os.environ)])

overwrite = True
error_score = np.nan
evaluation = WithinSessionEvaluation(paradigm=bench_cfg['paradigm'], datasets=bench_cfg['dataset'],
                                     suffix=unique_suffix, overwrite=overwrite, random_state=8, error_score=error_score)
results = evaluation.process(pipelines)
result_path = RESULTS_FOLDER / f'{identifier}_results.csv'

results.to_csv(result_path, encoding='utf-8', index=False)
t1 = time.time()
print(f'Benchmark run completed. Elapsed time: {(t1-t0)/3600} hours.')
