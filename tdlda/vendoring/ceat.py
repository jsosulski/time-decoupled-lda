import inspect

import sklearn.pipeline

from sklearn.base import clone
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory
import pyriemann.tangentspace
import pyriemann.classification
import pyriemann.utils.mean
import numpy as np


# based on scikit-learn version 0.21.3
class SamplePropsPipeline(sklearn.pipeline.Pipeline):
    def __init__(self, steps, memory=None, verbose=False,
                 sample_props_param_name='sample_props', require_sample_props_param=False):
        super(SamplePropsPipeline, self).__init__(steps, memory, verbose)
        self.sample_props_param_name = sample_props_param_name
        self.require_sample_props_param = require_sample_props_param

    def _get_stepwise_sample_prop_params(self, params, estimator_method):
        # create dict mapping step name -> parameter name -> parameter value
        # this returns only parameters for estimators that expect the (or all) keywords
        if self.sample_props_param_name not in params:
            if self.require_sample_props_param:
                raise ValueError(f'expected {self.sample_props_param_name} parameter, got {params}')

        stepwise_params = {}

        for cur_step_name in self.named_steps:
            stepwise_params[cur_step_name] = dict()
            cur_step = self.named_steps[cur_step_name]
            if hasattr(cur_step, estimator_method):
                cur_arg_spec = inspect.getfullargspec(getattr(cur_step, estimator_method))
                if (self.sample_props_param_name in cur_arg_spec.args
                        or self.sample_props_param_name in cur_arg_spec.kwonlyargs
                        # we don't support catch-all keyword params since those are used, e.g.,
                        # in the TransformerMixin for estimators that don't support keywords
                        # or cur_arg_spec.varkw is not None
                ):
                    stepwise_params[cur_step_name][self.sample_props_param_name] = params[
                        self.sample_props_param_name]

        return stepwise_params

    def _fit(self, X, y=None, **fit_params):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        # extract sample props params and remove them from the stepwise fit_params
        stepwise_fit_sp_params = self._get_stepwise_sample_prop_params(fit_params, 'fit')
        stepwise_transform_sp_params = self._get_stepwise_sample_prop_params(
            fit_params, 'transform')
        stepwise_fit_transform_sp_params = self._get_stepwise_sample_prop_params(
            fit_params, 'fit_transform')
        fit_params = {k: v for k, v in fit_params.items() if k != self.sample_props_param_name}

        fit_params_steps = {name: {} for name, step in self.steps
                            if step is not None}
        for pname, pval in fit_params.items():
            if '__' not in pname:
                raise ValueError(
                    "Pipeline.fit does not accept the {} parameter. "
                    "You can pass parameters to specific steps of your "
                    "pipeline using the stepname__parameter format, e.g. "
                    "`Pipeline.fit(X, y, logisticregression__sample_weight"
                    "=sample_weight)`.".format(pname))
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for (step_idx,
             name,
             transformer) in self._iter(with_final=False,
                                        filter_passthrough=False):
            if transformer is None or transformer == 'passthrough':
                with _print_elapsed_time('Pipeline',
                                         self._log_message(step_idx)):
                    continue

            if hasattr(memory, 'location'):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            elif hasattr(memory, 'cachedir'):
                # joblib < 0.11
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transfomer
            Xt, fitted_transformer = fit_transform_one_cached(
                cloned_transformer, Xt, y, None,
                message_clsname='Pipeline',
                message=self._log_message(step_idx),
                fit_sample_props_params=stepwise_fit_sp_params[name],
                transform_sample_props_params=stepwise_transform_sp_params[name],
                fit_transform_sample_props_params=stepwise_fit_transform_sp_params[name],
                **fit_params_steps[name]
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        if self._final_estimator == 'passthrough':
            return Xt, {}
        final_step_params = fit_params_steps[self.steps[-1][0]]
        final_step_params.update(stepwise_fit_sp_params[self.steps[-1][0]])
        return Xt, final_step_params

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, **predict_params):
        """Apply transforms to the data, and predict with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

        Returns
        -------
        y_pred : array-like
        """
        # extract sample props params and remove them from the final predict_params
        return self._call_prediction_method('predict', X, predict_params)

    def _transform(self, X, **transform_params):
        return self._transform_to_final(X, transform_params, with_final=True)

    def _transform_to_final(self, X, transform_params, with_final=False):
        stepwise_transform_sp_params = self._get_stepwise_sample_prop_params(transform_params,
                                                                             'transform')

        Xt = X
        for _, name, transform in self._iter(with_final=with_final):
            Xt = transform.transform(Xt, **stepwise_transform_sp_params[name])
        return Xt

    @if_delegate_has_method(delegate='_final_estimator')
    def fit_predict(self, X, y=None, **fit_params):
        """Applies fit_predict of last step in pipeline after transforms.

        Applies fit_transforms of a pipeline to the data, followed by the
        fit_predict method of the final estimator in the pipeline. Valid
        only if the final estimator implements fit_predict.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : array-like
        """

        # _fit already handles sample_props in fit_params, returns reduced set
        Xt, fit_params = self._fit(X, y, **fit_params)

        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][-1].fit_predict(Xt, y, **fit_params)
        return y_pred

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X, **predict_params):
        """Apply transforms, and predict_proba of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        return self._call_prediction_method('predict_proba', X, predict_params)

    def _call_prediction_method(self, predict_method, X, predict_params, y=None):
        Xt = self._transform_to_final(X, predict_params)

        predict_sample_props = self._get_stepwise_sample_prop_params(predict_params, predict_method)[
            self.steps[-1][0]]
        predict_params = {k: v for k, v in predict_params.items()
                          if k != self.sample_props_param_name}
        # include sample props (if needed for final step)

        predict_params.update(predict_sample_props)
        if y is None:
            return getattr(self.steps[-1][-1], predict_method)(Xt, **predict_params)
        else:  # for score score
            return getattr(self.steps[-1][-1], predict_method)(Xt, y, **predict_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def decision_function(self, X, **predict_params):
        """Apply transforms, and decision_function of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        return self._call_prediction_method('decision_function', X, predict_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_log_proba(self, X, **predict_params):
        """Apply transforms, and predict_log_proba of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        return self._call_prediction_method('predict_log_proba', X, predict_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None, sample_weight=None, **predict_params):
        """Apply transforms, and score with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
        """
        if sample_weight is not None:
            predict_params['sample_weight'] = sample_weight
        return self._call_prediction_method('score', X=X, y=y, predict_params=predict_params)

    def fit_transform(self, X, y=None, **fit_params):
        stepwise_transform_sp_params = self._get_stepwise_sample_prop_params(fit_params,
                                                                             'transform')
        last_step = self._final_estimator
        Xt, last_fit_params = self._fit(X, y, **fit_params)
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            if last_step == 'passthrough':
                return Xt
            if hasattr(last_step, 'fit_transform'):
                return last_step.fit_transform(Xt, y, **last_fit_params)
            else:
                return last_step.fit(Xt, y, **last_fit_params).transform(
                    Xt, **stepwise_transform_sp_params[self.steps[-1][0]])


def _fit_transform_one(transformer,
                       X,
                       y,
                       weight,
                       message_clsname='',
                       message=None,
                       fit_sample_props_params=None,
                       transform_sample_props_params=None,
                       fit_transform_sample_props_params=None,
                       **fit_params):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    if fit_sample_props_params is None:
        fit_sample_props_params = dict()
    if transform_sample_props_params is None:
        transform_sample_props_params = dict()
    if fit_transform_sample_props_params is None:
        fit_transform_sample_props_params = dict()
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, 'fit_transform'):
            res = transformer.fit_transform(X, y, **fit_params, **fit_transform_sample_props_params)
        else:
            res = transformer.fit(X, y, **fit_params, **fit_sample_props_params).transform(
                X, **transform_sample_props_params)

    if weight is None:
        return res, transformer
    return res * weight, transformer


class VariableReferenceTangentSpace(pyriemann.tangentspace.TangentSpace):
    def __init__(self, metric='riemann', tsupdate=False,
                 tangent_space_reference='mean', random_seed=None):
        super(VariableReferenceTangentSpace, self).__init__(metric=metric,
                                                            tsupdate=tsupdate)
        self.tangent_space_reference = tangent_space_reference
        self.random_seed = random_seed

    def fit(self, X, y=None, sample_weight=None):
        random = np.random.RandomState(self.random_seed)
        if self.tangent_space_reference == 'mean':
            self.reference_ = pyriemann.utils.mean.mean_covariance(X, metric=self.metric,
                                              sample_weight=sample_weight)
        elif self.tangent_space_reference == 'random':
            # when using this option, multiple repetitions should be made...
            ref_sample_idx = random.randint(len(X))
            self.reference_ = X[ref_sample_idx]
        elif self.tangent_space_reference == 'identity':
            self.reference_ = np.eye(*X.shape[1:])
        elif self.tangent_space_reference.startswith('fraction_'):
            # e.g., fraction_0.1
            sample_size = int(float(self.tangent_space_reference.replace('fraction_', '')) * len(X))
            candidate_idxs = np.arange(len(X))
            random.shuffle(candidate_idxs)
            ref_sample_idxs = candidate_idxs[:sample_size]
            self.reference_ = pyriemann.utils.mean.mean_covariance(
                X[ref_sample_idxs], metric=self.metric,
                sample_weight=sample_weight)
        else:
            raise ValueError('unknown tangent space reference strategy {}'.format(
                self.tangent_space_reference))

    def fit_transform(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.transform(X)


def create_logistic_regression(penalty='l2'):
    penalty = penalty  # TODO l1 or l2
    class_priors = 'balanced'
    if class_priors == 'uniform':
        class_priors = None

    return sklearn.linear_model.LogisticRegression(
        penalty=penalty,
        class_weight=class_priors,
        solver='liblinear',
        random_state=42)  # seed for liblinear (data shuffling!?)


