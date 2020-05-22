from typing import Tuple
import numpy as np
import sklearn
import sklearn.utils.multiclass
import sklearn.linear_model

from sklearn.preprocessing import StandardScaler


def diag_indices_with_offset(p, offset):
    idxdiag = np.diag_indices(p)
    idxdiag_with_offset = list()
    idxdiag_with_offset.append(np.array([i + offset for i in idxdiag[0]]))
    idxdiag_with_offset.append(np.array([i + offset for i in idxdiag[1]]))
    return tuple(idxdiag_with_offset)


def _shrinkage(X: np.ndarray, gamma=None, T=None, S=None, block=False,
               N_channels=31, N_times=5, standardize=True) -> Tuple[np.ndarray, float]:

    p, n = X.shape

    if standardize:
        sc = StandardScaler()  # standardize_featurestd features
        X = sc.fit_transform(X.T).T
    Xn = X - np.repeat(np.mean(X, axis=1, keepdims=True), n, axis=1)
    if S is None:
        S = np.matmul(Xn, Xn.T)
    Xn2 = np.square(Xn)
    idxdiag = np.diag_indices(p)

    # Target = B
    nu = np.mean(S[idxdiag])
    if T is None:
        if block:
            nu = list()
            for i in range(N_times):
                idxblock = diag_indices_with_offset(N_channels, i*N_channels)
                nu.append([np.mean(S[idxblock])] * N_channels)
            nu = [sl for l in nu for sl in l]
            T = np.diag(np.array(nu))
        else:
            T = nu * np.eye(p, p)

    # Ledoit Wolf
    V = 1. / (n - 1) * (np.matmul(Xn2, Xn2.T) - np.square(S) / n)
    if gamma is None:
        gamma = n * np.sum(V) / np.sum(np.square(S - T))
    if gamma > 1:
        print("logger.warning('forcing gamma to 1')")
        gamma = 1
    elif gamma < 0:
        print("logger.warning('forcing gamma to 0')")
        gamma = 0

    Cstar = (gamma * T + (1 - gamma) * S) / (n - 1)
    if standardize:  # scale back
        Cstar = sc.scale_[np.newaxis, :] * Cstar * sc.scale_[:, np.newaxis]
    return Cstar, gamma


# corresponds to train_RLDAshrink.m and clsutil_shrinkage.m from bbci_public
class ShrinkageLinearDiscriminantAnalysis(
        sklearn.base.BaseEstimator,
        sklearn.linear_model.base.LinearClassifierMixin):

    def __init__(self, priors=None, only_block=False, N_times=5, N_channels=31, pool_cov=True, standardize_shrink=True):
        self.only_block = only_block
        self.priors = priors
        self.N_times = N_times
        self.N_channels = N_channels
        self.pool_cov = pool_cov
        self.standardize_shrink = standardize_shrink

    def fit(self, X_train, y):
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        if set(self.classes_) != {0, 1}:
            raise ValueError('currently only binary class supported')
        assert len(X_train) == len(y)
        xTr = X_train.T

        n_classes = 2
        if self.priors is None:
            # here we deviate from the bbci implementation and
            # use the sample priors by default
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            priors = np.bincount(y_t) / float(len(y))
            # self.priors = np.array([1./n_classes] * n_classes)
        else:
            priors = self.priors

        X, cl_mean = subtract_classwise_means(xTr, y)
        if self.pool_cov:
            C_cov, C_gamma = _shrinkage(X, N_channels=self.N_channels, N_times=self.N_times,
                                        standardize=self.standardize_shrink)
        else:
            n_classes = 2
            C_cov = np.zeros((xTr.shape[0], xTr.shape[0]))
            for cur_class in range(n_classes):
                class_idxs = y == cur_class
                x_slice = X[:, class_idxs]
                C_cov += priors[cur_class] * _shrinkage(x_slice)[0]

        if self.only_block:
            C_cov_new = np.zeros_like(C_cov)
            for i in range(self.N_times):
                idx_start = i * self.N_channels
                idx_end = idx_start + self.N_channels
                C_cov_new[idx_start:idx_end, idx_start:idx_end] = C_cov[idx_start:idx_end, idx_start:idx_end]
            C_cov = C_cov_new

        C_invcov = np.linalg.pinv(C_cov)
        # w = np.matmul(C_invcov, cl_mean)
        w = np.linalg.lstsq(C_cov, cl_mean)[0]
        b = -0.5 * np.sum(cl_mean * w, axis=0).T + np.log(priors)

        if n_classes == 2:
            w = w[:, 1] - w[:, 0]
            b = b[1] - b[0]

        self.coef_ = w.reshape((1, -1))
        self.intercept_ = b

    def predict_proba(self, X):
        """Estimate probability.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated probabilities.
        """
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)

        return np.column_stack([1 - prob, prob])

    def predict_log_proba(self, X):
        """Estimate log probability.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        return np.log(self.predict_proba(X))


class TimeDecoupledLda(
        ShrinkageLinearDiscriminantAnalysis):
    """shrinkage LdaClasswiseCovs with enhancement technique for ERP classification

    Parameters
    ----------
    inverted : bool (default: False)
        If you want to estimate and change the diagonal blocks before
        inverting the covariance matrix.
    plot: bool (default: False)
        If you want to plot the original covariance matrix,
        the new diagonal box and the new matrix.
    """

    def __init__(self, priors=None, N_times=5, N_channels=31, standardize_featurestd=False, preproc=None,
                 standardize_shrink=True, channel_gamma=None):
        self.priors = priors
        self.N_times = N_times
        self.N_channels = N_channels
        self.standardize_featurestd = standardize_featurestd
        self.standardize_shrink = standardize_shrink
        self.channel_gamma = channel_gamma
        self.preproc = preproc  # This is needed to obtain time interval standardization factors from vectorizer

    def fit(self, X_train, y):
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        if set(self.classes_) != {0, 1}:
            raise ValueError('currently only binary class supported')
        assert len(X_train) == len(y)
        xTr = X_train.T

        if self.priors is None:
            # here we deviate from the bbci implementation and
            # use the sample priors by default
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            priors = np.bincount(y_t) / float(len(y))
        else:
            priors = self.priors

        X, cl_mean = subtract_classwise_means(xTr, y)  # outsourced to method
        C_cov, C_gamma = _shrinkage(X, N_channels=self.N_channels, N_times=self.N_times,
                                    standardize=self.standardize_shrink)
        C_cov = change_diagonal_entries(C_cov, xTr, y, inverted=False, N_times=self.N_times,
                                        N_channels=self.N_channels, standardize=self.standardize_featurestd,
                                        jumping_means_ivals=self.preproc.jumping_mean_ivals,
                                        channel_gamma=self.channel_gamma)

        w = np.linalg.lstsq(C_cov, cl_mean)[0]
        b = -0.5 * np.sum(cl_mean * w, axis=0).T + np.log(priors)
        w = w[:, 1] - w[:, 0]
        b = b[1] - b[0]

        self.coef_ = w.reshape((1, -1))
        self.intercept_ = b


def subtract_classwise_means(xTr, y):
    n_classes = 2
    n_features = xTr.shape[0]
    X = np.zeros((n_features, 0))
    cl_mean = np.zeros((n_features, n_classes))
    for cur_class in range(n_classes):
        class_idxs = y == cur_class
        cl_mean[:, cur_class] = np.mean(xTr[:, class_idxs], axis=1)

        X = np.concatenate([
            X,
            xTr[:, class_idxs] - np.dot(cl_mean[:, cur_class].reshape(-1, 1),
                                        np.ones((1, np.sum(class_idxs))))],
               axis=1)
    return X, cl_mean


def change_diagonal_entries(S, xTr, y, inverted=False, N_times=5, N_channels=31, standardize=False,
                            jumping_means_ivals=None, channel_gamma=None):
    # compute sigma_c
    # information about time not relevant
    if standardize:
        if jumping_means_ivals is not None:
            num_samples = ((np.diff(np.array(jumping_means_ivals))+0.001)/0.01).squeeze()
            factors = np.sqrt(num_samples / np.min(num_samples))
            for ti in range(N_times):
                start_i = N_channels * ti
                end_i = start_i + N_channels
                xTr[start_i:end_i, :] *= factors[ti]
    xTr_meanfree, class_means = subtract_classwise_means(xTr, y)
    X_long_slim = xTr_meanfree.reshape((N_channels, -1), order='F')

    sigma_c_ref, gamma_c = _shrinkage(X_long_slim, N_channels=N_channels, N_times=N_times, standardize=True,
                                      gamma=channel_gamma)
    sigma_c = np.linalg.pinv(sigma_c_ref) if inverted else sigma_c_ref
    sign_sigma_c, slogdet_sigma_c = np.linalg.slogdet(sigma_c)
    logdet_sigma_c = slogdet_sigma_c * sign_sigma_c
    # compute scalar to scale sigma_c and change diagonal boxes of cov.matrix

    S_new = np.copy(S)

    for i in range(N_times):
        idx_start = i*N_channels
        idx_end = idx_start + N_channels
        sigma_block = S[idx_start:idx_end, idx_start:idx_end]
        sign_sigma_block, slogdet_sigma_block = np.linalg.slogdet(sigma_block)
        logdet_sigma_block = slogdet_sigma_block * sign_sigma_block
        scalar_via_determinant = np.exp(logdet_sigma_block - logdet_sigma_c)**(1.0/S[idx_start:idx_end, idx_start:idx_end].shape[1])
        if scalar_via_determinant < 1:
            pass
        S_new[idx_start:idx_end, idx_start:idx_end] = sigma_c * scalar_via_determinant  # * scaling_factor
    S = S_new

    if np.any(np.isnan(S)) or np.any(np.isinf(S)):
        raise OverflowError('Diagonal-block covariance matrix is not numeric.')
    return S



