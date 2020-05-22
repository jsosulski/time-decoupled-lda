from tdlda.benchmark.feature_preprocessing import Vectorizer
from tdlda.classification.bbci_lda import _shrinkage
import numpy as np


def time_ivals_to_idx(epo, ivals):
    idxs = []
    for i, ival in enumerate(ivals):
        idx = epo.time_as_index((ival[0], ival[1]), use_rounding=True)
        idx_range = list(range(idx[0], idx[1]))
        idxs.append(idx_range)
    return idxs


def epoch_to_common_covariance(epo, ivals, pooled_covariance=True, ch_picks='eeg', unit_factor=1e6):
    epo = epo.pick(ch_picks)
    jm_vec = Vectorizer(jumping_mean_ivals=ivals)
    n_ivals = len(ivals)
    n_channels = len(epo.ch_names)
    n_epos = len(epo)
    classwise_arr = list()
    epos_per_class = np.zeros((len(epo.event_id,)))

    for i, ev in enumerate(epo.event_id):
        epos_per_class[i] = len(epo[ev])
        x = epo[ev].get_data()
        x = jm_vec.transform(x * unit_factor)
        if pooled_covariance:
            x_meanfree = x - np.mean(x, axis=0, keepdims=True)
            classwise_arr.append(x_meanfree)
        else:
            sigma = _shrinkage(x.T)[0]
            classwise_arr.append(sigma)

    if pooled_covariance:
        sigma = _shrinkage(np.concatenate(classwise_arr).T)[0]
    else:
        sigma = np.zeros((n_channels*n_ivals, n_channels*n_ivals))
        for i, sigma_cl in enumerate(classwise_arr):
            weight_factor = epos_per_class[i] / n_epos
            sigma += weight_factor * sigma_cl

    return sigma