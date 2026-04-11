"""Utilities for running a single experiment configuration."""

import numpy as np

from project1.data.loaders import load_dataset
from project1.data.preprocessing import prepare_model_ready_splits
from project1.data.split import make_data_split
from project1.metrics.classification import score_binary_classification
from project1.missingness.generators import MAR1, MAR2, MCAR, MNAR
from project1.models.baselines import NaiveLogReg, OracleLogReg
from project1.models.unlabeled_logreg import UnlabeledLogReg


_MISSINGNESS_GENERATORS = {
    "mcar": MCAR,
    "mar1": MAR1,
    "mar2": MAR2,
    "mnar": MNAR,
}


def _to_probability_scores(model, X):
    """Return positive-class scores for metric computation."""
    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Model predict_proba must return an array of shape (n_samples, 2).")
    return probabilities[:, 1]


def _compute_test_metrics(y_true, y_pred, y_score):
    """Compute the test metrics required by the single-experiment runner."""
    return {
        "accuracy": score_binary_classification(y_true, y_pred, y_score, measure="accuracy"),
        "balanced_accuracy": score_binary_classification(y_true, y_pred, y_score, measure="balanced_accuracy"),
        "f1": score_binary_classification(y_true, y_pred, y_score, measure="f1"),
        "roc_auc": score_binary_classification(y_true, y_pred, y_score, measure="auc"),
    }


def _generate_missing_train_labels(X_train, y_train, missingness, missing_proba, random_state, **missingness_kwargs):
    """Generate missing labels in the training split only."""
    if missingness not in _MISSINGNESS_GENERATORS:
        valid_names = ", ".join(sorted(_MISSINGNESS_GENERATORS))
        raise ValueError(f"Unsupported missingness mechanism: {missingness}. Valid options: {valid_names}")

    generator = _MISSINGNESS_GENERATORS[missingness]
    return generator(X_train, y_train, missing_proba, seed=random_state, **missingness_kwargs).astype(int)


def _build_model(method, random_state, label_completion_method="logistic", **model_kwargs):
    """Instantiate a supported experiment model."""
    if method == "naive":
        return NaiveLogReg(random_state=random_state, **model_kwargs)
    if method == "oracle":
        return OracleLogReg(random_state=random_state, **model_kwargs)
    if method == "unlabeled":
        return UnlabeledLogReg(
            imputation_type=label_completion_method,
            random_state=random_state,
            **model_kwargs,
        )

    valid_methods = "naive, oracle, unlabeled"
    raise ValueError(f"Unsupported method: {method}. Valid options: {valid_methods}")


def run_single_experiment(
    dataset,
    scheme,
    method,
    missing_rate,
    seed=42,
    label_completion_method="logistic",
    test_size=0.2,
    valid_size=0.2,
    correlation_threshold=0.95,
    missingness_kwargs=None,
    model_kwargs=None,
):
    """Run one experiment from data loading through test-set evaluation.

    Parameters
    ----------
    dataset : str
        Dataset name supported by `load_dataset`.
    scheme : str
        One of `mcar`, `mar1`, `mar2`, or `mnar`.
    method : str
        One of `naive`, `oracle`, or `unlabeled`.
    missing_rate : float
        Fraction or average probability of missing labels in train.
    seed : int, default=42
        Random seed used in splitting, missingness, and model initialization.
    label_completion_method : str, default="logistic"
        Imputation strategy passed to `UnlabeledLogReg` when `method="unlabeled"`.
        For other methods the value is ignored and stored as `None` in results.
    test_size : float, default=0.2
        Fraction of data assigned to the test split.
    valid_size : float, default=0.2
        Fraction of data assigned to the validation split.
    correlation_threshold : float, default=0.95
        Threshold for dropping highly correlated features during preprocessing.
    missingness_kwargs : dict or None
        Extra keyword arguments forwarded to the missingness generator.
    model_kwargs : dict or None
        Extra keyword arguments forwarded to the selected model constructor.

    Returns
    -------
    dict
        Single-row experiment result with metadata and test metrics.
    """
    scheme = scheme.lower()
    method = method.lower()
    if label_completion_method is None:
        label_completion_method = "logistic"
    label_completion_method = label_completion_method.lower()

    if label_completion_method not in {"logistic", "knn", "prior"}:
        raise ValueError("label_completion_method must be one of: logistic, knn, prior.")

    effective_label_completion_method = label_completion_method if method == "unlabeled" else None

    missingness_kwargs = {} if missingness_kwargs is None else dict(missingness_kwargs)
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

    if scheme == "mar1":
        missingness_kwargs.setdefault("missing_influence_col_index", 0)

    X, y, feature_names = load_dataset(dataset)
    X_train, X_valid, X_test, y_train, y_valid, y_test = make_data_split(
        X,
        y,
        random_state=seed,
        test_size=test_size,
        valid_size=valid_size,
    )
    prepared = prepare_model_ready_splits(
        X_train,
        y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        correlation_threshold=correlation_threshold,
    )

    y_train_obs = _generate_missing_train_labels(
        prepared.X_train,
        prepared.y_train,
        missingness=scheme,
        missing_proba=missing_rate,
        random_state=seed,
        **missingness_kwargs,
    )

    model = _build_model(
        method,
        random_state=seed,
        label_completion_method=label_completion_method,
        **model_kwargs,
    )

    if method == "naive":
        model.fit(prepared.X_train, y_train_obs)
    elif method == "oracle":
        model.fit(prepared.X_train, prepared.y_train)
    else:
        model.fit(prepared.X_train, y_train_obs, X_validate=prepared.X_valid, y_validate=prepared.y_valid)

    y_test_pred = model.predict(prepared.X_test)
    y_test_score = _to_probability_scores(model, prepared.X_test)
    metrics = _compute_test_metrics(prepared.y_test, y_test_pred, y_test_score)

    observed_fraction = float(np.mean(y_train_obs != -1))
    result = {
        "dataset": dataset,
        "scheme": scheme,
        "method": method,
        "label_completion_method": effective_label_completion_method,
        "seed": seed,
        "missing_rate": missing_rate,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "train_size": int(prepared.X_train.shape[0]),
        "valid_size": int(prepared.X_valid.shape[0]),
        "test_size": int(prepared.X_test.shape[0]),
        "n_features_before_preprocessing": int(len(feature_names)),
        "n_features_after_preprocessing": int(len(prepared.feature_names)),
        "observed_label_fraction_train": observed_fraction,
    }

    return result


__all__ = ["run_single_experiment"]
