from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class Vectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, permute_channels_and_time=True, jumping_mean_ivals=None,
                 select_ival=None, fs=100, averaging_samples=None):
        self.permute_channels_and_time = permute_channels_and_time
        self.jumping_mean_ivals = jumping_mean_ivals
        self.select_ival = select_ival
        self.fs = fs
        self.averaging_samples = averaging_samples
        if select_ival is not None and jumping_mean_ivals is not None:
            raise ValueError('Cannot both calculate jumping means and select all samples in an ival.')
        if select_ival is None and jumping_mean_ivals is None:
            raise ValueError('Choose either a select_ival or a jumping_means_ival.')

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """
        fs = self.fs
        if self.jumping_mean_ivals is not None:
            self.averaging_samples = np.zeros(len(self.jumping_mean_ivals))
            new_X = np.zeros((X.shape[0], X.shape[1], len(self.jumping_mean_ivals)))
            for i, ival in enumerate(self.jumping_mean_ivals):
                np_idx = np.round(np.arange(ival[0], ival[1], 1/fs) * fs).astype(int)
                if np_idx.shape != np.unique(np_idx).shape:
                    print('WARNING: duplicate elements in np_idx')
                idx = np.unique(np_idx).tolist()
                self.averaging_samples[i] = len(idx)
                new_X[:, :, i] = np.mean(X[:, :, idx], axis=2)
        elif self.select_ival is not None:
            np_idx = np.round(np.arange(self.select_ival[0], self.select_ival[1], 1/fs) * fs).astype(int)
            if np_idx.shape != np.unique(np_idx).shape:
                print('WARNING: duplicate elements in np_idx')
            idx = np.unique(np_idx).tolist()
            new_X = X[:, :, idx]
        else:
            assert False, 'In the constructor, pass either select ival or jumping means.'
        X = new_X
        if self.permute_channels_and_time:
            X = X.transpose((0, 2, 1))
        X = np.reshape(X, (X.shape[0], -1))
        return X
