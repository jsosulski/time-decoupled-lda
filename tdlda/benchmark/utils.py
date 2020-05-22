from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LdaClasswiseCovs
from sklearn.decomposition.kernel_pca import KernelPCA

from tdlda.classification.bbci_lda import ShrinkageLinearDiscriminantAnalysis as LdaPooledCovs
from tdlda.classification.bbci_lda import TimeDecoupledLda as LdaImpPooledCov

from tdlda.benchmark.feature_preprocessing import Vectorizer

from moabb.paradigms import P300
from moabb.datasets import EPFLP300, bi2013a
from moabb.datasets import BNCI2014009 as bnci_1
from moabb.datasets import BNCI2014008 as bnci_als
from moabb.datasets import BNCI2015003 as bnci_2
from tdlda.datasets.spot_pilot import SpotPilotData

import pyriemann
import numpy as np

from tdlda.vendoring.ceat import SamplePropsPipeline, VariableReferenceTangentSpace, create_logistic_regression

def create_only_best_paper_pipelines(cfg, feature_preprocessing_key, N_channels, labels_dict):
    pipelines = dict()
    # Add Riemann pipeline
    n_xdawn_components = 5
    xdawn_class = 'Target'
    ts_metric = 'riemann'
    riemann_cov_type = 'lwf'
    xdawn_cov_type = 'scm'
    xdawn_classes_ = [labels_dict[xdawn_class]]
    new_key = f'ceat_rg_xdawncomps_{n_xdawn_components}_xdawnclasses_{xdawn_class}'
    pipelines[new_key] = SamplePropsPipeline([
                ('xdawn',
                 pyriemann.estimation.XdawnCovariances(nfilter=n_xdawn_components,
                                                       classes=xdawn_classes_,
                                                       estimator=riemann_cov_type,
                                                       xdawn_estimator=xdawn_cov_type,
                                                       applyfilters=True)),
                ('TangentSpace', VariableReferenceTangentSpace(
                    metric=ts_metric, tsupdate=False,
                    tangent_space_reference='mean',
                    random_seed=np.random.randint(2 ** 32 - 1))),
                ('LogisticRegression', create_logistic_regression(penalty='l2'))
            ])
    # Add all three LDA versions
    fs = cfg['default']['data_preprocessing']['sampling_rate']
    cfg_vect = cfg['default'][feature_preprocessing_key]['feature_preprocessing']
    c_jm = cfg_vect['jumping_means_ival']
    c_sel = cfg_vect['select_ival']
    vectorizers = dict()
    key = 'numerous'
    vectorizers[f'jm_{key}'] = dict(vec=Vectorizer(jumping_mean_ivals=c_jm[key]['ival']), D=c_jm[key]['D'], fs=fs)
    classifiers = dict(
        lda_c_covs=LdaClasswiseCovs(solver='lsqr', shrinkage='auto'),
        lda_p_cov=LdaPooledCovs(N_channels=N_channels),
        lda_imp_p_cov=LdaImpPooledCov(N_channels=N_channels, standardize_featurestd=True, channel_gamma=0),
    )
    for v_key in vectorizers.keys():
        D = vectorizers[v_key]['D']
        vec = vectorizers[v_key]['vec']
        for c_key in classifiers.keys():
            clf = clone(classifiers[c_key])
            clf.N_times = D
            new_key = f'{v_key}_{c_key}'
            clf.preproc = vec
            pipelines[new_key] = make_pipeline(vec, clf)
        ncomp = 70
        pipelines[f'{v_key}_kPCA({ncomp})_skl_lsqr'] = make_pipeline(vec, KernelPCA(n_components=ncomp),
                                                                         LdaClasswiseCovs(solver='lsqr', shrinkage='auto'))
    return pipelines

def create_ceat_rg_lr_pipelines(labels_dict):
    pipelines = dict()
    # loop over these
    n_xdawn_components = [1, 2, 3, 4, 5, 6]
    xdawn_classes = ['both', 'Target']

    # fixed
    ts_metric = 'riemann'
    riemann_cov_type = 'lwf'
    xdawn_cov_type = 'scm'
    for nxc in n_xdawn_components:
        for xc in xdawn_classes:
            xdawn_classes_ = None if xc == 'both' else [labels_dict[xc]]
            new_key = f'ceat_rg_xdawncomps_{nxc}_xdawnclasses_{xc}'
            pipelines[new_key] = SamplePropsPipeline([
                ('xdawn',
                 pyriemann.estimation.XdawnCovariances(nfilter=nxc,
                                                       classes=xdawn_classes_,
                                                       estimator=riemann_cov_type,
                                                       xdawn_estimator=xdawn_cov_type,
                                                       applyfilters=True)),
                ('TangentSpace', VariableReferenceTangentSpace(
                    metric=ts_metric, tsupdate=False,
                    tangent_space_reference='mean',
                    random_seed=np.random.randint(2 ** 32 - 1))),
                ('LogisticRegression', create_logistic_regression(penalty='l2'))
            ])
    return pipelines


def create_lda_pipelines(cfg, feature_preprocessing_key, N_channels):
    pipelines = dict()
    fs = cfg['default']['data_preprocessing']['sampling_rate']
    cfg_vect = cfg['default'][feature_preprocessing_key]['feature_preprocessing']
    c_jm = cfg_vect['jumping_means_ival']
    c_sel = cfg_vect['select_ival']
    vectorizers = dict()
    for key in c_jm:
        vectorizers[f'jm_{key}'] = dict(vec=Vectorizer(jumping_mean_ivals=c_jm[key]['ival']), D=c_jm[key]['D'], fs=fs)
    for key in c_sel:
        vectorizers[f'sel_{key}'] = dict(vec=Vectorizer(select_ival=c_sel[key]['ival']), D=c_sel[key]['D'], fs=fs)
    classifiers = dict(
        lda_c_covs=LdaClasswiseCovs(solver='lsqr', shrinkage='auto'),
        lda_p_cov=LdaPooledCovs(N_channels=N_channels),
        lda_imp_p_cov=LdaImpPooledCov(N_channels=N_channels, standardize_featurestd=True, channel_gamma=0),
    )
    for v_key in vectorizers.keys():
        D = vectorizers[v_key]['D']
        vec = vectorizers[v_key]['vec']
        for c_key in classifiers.keys():
            clf = clone(classifiers[c_key])
            clf.N_times = D
            new_key = f'{v_key}_{c_key}'
            clf.preproc = vec  # why does this not persist :(
            pipelines[new_key] = make_pipeline(vec, clf)
    for v_key in vectorizers.keys():
        D = vectorizers[v_key]['D']
        vec = vectorizers[v_key]['vec']
        for c_key in classifiers.keys():
            clf = clone(classifiers[c_key])
            clf.N_times = D
            new_key = f'{v_key}_{c_key}'
            clf.preproc = vec  # why does this not persist :(
            pipelines[new_key] = make_pipeline(vec, clf)
        for i in range(1, 11):
            ncomp = i * 10 if i < 10 else None
            pipelines[f'{v_key}_kPCA({ncomp})_skl_lsqr'] = make_pipeline(vec, KernelPCA(n_components=ncomp),
                                                                         LdaClasswiseCovs(solver='lsqr', shrinkage='auto'))
    return pipelines


def get_benchmark_config(dataset_name, cfg_prepro, subjects=None, sessions=None):
    benchmark_cfg = dict()
    paradigm = P300(resample=cfg_prepro['sampling_rate'], fmin=cfg_prepro['fmin'], fmax=cfg_prepro['fmax'],
                    reject_uv=cfg_prepro['reject_uv'], baseline=cfg_prepro['baseline'])
    load_ival = [0, 1]
    if dataset_name == 'spot_single':
        d = SpotPilotData(load_single_trials=True)
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = d.N_channels
    elif dataset_name == 'epfl':
        d = EPFLP300()
        d.interval = load_ival
        d.unit_factor = 1
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 32
    elif dataset_name == 'bnci_1':
        d = bnci_1()
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 16
    elif dataset_name == 'bnci_als':
        d = bnci_als()
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 8
    elif dataset_name == 'bnci_2':
        d = bnci_2()
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 8
    elif dataset_name == 'braininvaders':
        d = bi2013a()
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 16
    else:
        raise ValueError(f'Dataset {dataset_name} not recognized.')

    benchmark_cfg['dataset'] = d
    benchmark_cfg['N_channels'] = n_channels
    benchmark_cfg['paradigm'] = paradigm
    return benchmark_cfg
