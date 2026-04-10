"""Preprocessing utilities for Project 1."""

from dataclasses import dataclass

import numpy as np


def add_intercept_column(df):
    """Insert an intercept column at the beginning of a DataFrame."""
    df.insert(0, "intercept", 1)
    return df


def _to_numpy_features(X):
    """Convert feature input to a two-dimensional NumPy array."""
    if hasattr(X, "to_numpy"):
        X_array = X.to_numpy(dtype=float)
    else:
        X_array = np.asarray(X, dtype=float)

    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array or DataFrame.")
    return X_array


def _normalize_feature_names(feature_names, n_features):
    """Return a feature-name list aligned with the number of columns in `X`."""
    if feature_names is None:
        return [f"x{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError("feature_names length must match the number of columns in X.")
    return list(feature_names)


def fit_numeric_imputer(X):
    """Estimate per-column means for numeric missing-value imputation."""
    X_array = _to_numpy_features(X)
    fill_values = np.nanmean(X_array, axis=0)
    fill_values = np.where(np.isnan(fill_values), 0.0, fill_values)
    return fill_values


def transform_numeric_imputer(X, fill_values):
    """Impute missing numeric values using precomputed per-column fill values."""
    X_array = _to_numpy_features(X).copy()
    if X_array.shape[1] != len(fill_values):
        raise ValueError("fill_values length must match the number of columns in X.")

    missing_mask = np.isnan(X_array)
    if np.any(missing_mask):
        X_array[missing_mask] = np.take(fill_values, np.where(missing_mask)[1])
    return X_array


def fit_correlated_feature_filter(X, feature_names, correlation_threshold=0.95):
    """Select a subset of features by removing columns above the correlation threshold."""
    X_array = _to_numpy_features(X)
    normalized_feature_names = _normalize_feature_names(feature_names, X_array.shape[1])

    if not 0 <= correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be between 0 and 1.")

    with np.errstate(invalid="ignore", divide="ignore"):
        correlation_matrix = np.atleast_2d(np.corrcoef(X_array, rowvar=False))
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

    keep_mask = np.ones(X_array.shape[1], dtype=bool)
    for col_idx in range(1, X_array.shape[1]):
        if not keep_mask[col_idx]:
            continue
        correlated_with_kept = np.abs(correlation_matrix[:col_idx, col_idx]) >= correlation_threshold
        if np.any(correlated_with_kept & keep_mask[:col_idx]):
            keep_mask[col_idx] = False

    kept_indices = np.flatnonzero(keep_mask)
    kept_feature_names = [normalized_feature_names[idx] for idx in kept_indices]
    return kept_indices, kept_feature_names


def transform_correlated_feature_filter(X, kept_indices, feature_names=None):
    """Apply a previously fitted correlated-feature filter to `X`."""
    X_array = _to_numpy_features(X)
    if np.any(kept_indices >= X_array.shape[1]):
        raise ValueError("kept_indices contain values outside the column range of X.")

    filtered_X = X_array[:, kept_indices]
    filtered_feature_names = _normalize_feature_names(feature_names, X_array.shape[1])
    filtered_feature_names = [filtered_feature_names[idx] for idx in kept_indices]
    return filtered_X, filtered_feature_names


def fit_standardizer(X):
    """Estimate per-column means and standard deviations for numeric standardization."""
    X_array = _to_numpy_features(X)
    means = X_array.mean(axis=0)
    scales = X_array.std(axis=0)
    scales = np.where(scales == 0, 1.0, scales)
    return means, scales


def transform_standardizer(X, means, scales):
    """Standardize numeric data using precomputed train-set means and scales."""
    X_array = _to_numpy_features(X)
    if X_array.shape[1] != len(means) or X_array.shape[1] != len(scales):
        raise ValueError("means and scales must match the number of columns in X.")
    return (X_array - means) / scales


@dataclass
class NumericPreprocessingPipeline:
    """Preprocess numeric data with train-fitted imputation, correlation filtering, and scaling."""

    correlation_threshold: float = 0.95
    imputer_fill_values_: np.ndarray | None = None
    kept_feature_indices_: np.ndarray | None = None
    feature_names_: list[str] | None = None
    standardizer_means_: np.ndarray | None = None
    standardizer_scales_: np.ndarray | None = None

    def fit(self, X_train, feature_names=None):
        """Fit all preprocessing steps using training data only."""
        X_train_array = _to_numpy_features(X_train)
        initial_feature_names = _normalize_feature_names(feature_names, X_train_array.shape[1])

        self.imputer_fill_values_ = fit_numeric_imputer(X_train_array)
        X_train_imputed = transform_numeric_imputer(X_train_array, self.imputer_fill_values_)

        self.kept_feature_indices_, self.feature_names_ = fit_correlated_feature_filter(
            X_train_imputed,
            initial_feature_names,
            correlation_threshold=self.correlation_threshold,
        )
        X_train_filtered, _ = transform_correlated_feature_filter(
            X_train_imputed,
            self.kept_feature_indices_,
            feature_names=initial_feature_names,
        )

        self.standardizer_means_, self.standardizer_scales_ = fit_standardizer(X_train_filtered)
        return self

    def transform(self, X):
        """Transform a split with preprocessing parameters learned on the training set."""
        if self.imputer_fill_values_ is None or self.kept_feature_indices_ is None:
            raise ValueError("Pipeline is not fitted yet. Call fit or fit_transform first.")

        X_imputed = transform_numeric_imputer(X, self.imputer_fill_values_)
        X_filtered, _ = transform_correlated_feature_filter(X_imputed, self.kept_feature_indices_)
        X_standardized = transform_standardizer(X_filtered, self.standardizer_means_, self.standardizer_scales_)
        return X_standardized, list(self.feature_names_)

    def fit_transform(self, X_train, feature_names=None):
        """Fit the pipeline on train and return the transformed train split."""
        self.fit(X_train, feature_names=feature_names)
        return self.transform(X_train)


@dataclass
class PreparedDataSplit:
    """Container for model-ready train, validation, and test arrays."""

    X_train: np.ndarray
    X_valid: np.ndarray | None
    X_test: np.ndarray | None
    y_train: np.ndarray
    y_valid: np.ndarray | None
    y_test: np.ndarray | None
    feature_names: list[str]
    pipeline: NumericPreprocessingPipeline


def preprocess_train_valid_test(X_train, X_valid=None, X_test=None, feature_names=None, correlation_threshold=0.95):
    """Fit preprocessing on train and transform train, valid, and test without leakage."""
    pipeline = NumericPreprocessingPipeline(correlation_threshold=correlation_threshold)
    X_train_processed, final_feature_names = pipeline.fit_transform(X_train, feature_names=feature_names)

    X_valid_processed = None
    if X_valid is not None:
        X_valid_processed, _ = pipeline.transform(X_valid)

    X_test_processed = None
    if X_test is not None:
        X_test_processed, _ = pipeline.transform(X_test)

    return X_train_processed, X_valid_processed, X_test_processed, final_feature_names, pipeline


def prepare_model_ready_splits(
    X_train,
    y_train,
    X_valid=None,
    y_valid=None,
    X_test=None,
    y_test=None,
    feature_names=None,
    correlation_threshold=0.95,
):
    """Preprocess split data and return model-ready arrays plus final feature names."""
    X_train_processed, X_valid_processed, X_test_processed, final_feature_names, pipeline = preprocess_train_valid_test(
        X_train,
        X_valid=X_valid,
        X_test=X_test,
        feature_names=feature_names,
        correlation_threshold=correlation_threshold,
    )
    return PreparedDataSplit(
        X_train=X_train_processed,
        X_valid=X_valid_processed,
        X_test=X_test_processed,
        y_train=np.ravel(np.asarray(y_train)),
        y_valid=None if y_valid is None else np.ravel(np.asarray(y_valid)),
        y_test=None if y_test is None else np.ravel(np.asarray(y_test)),
        feature_names=final_feature_names,
        pipeline=pipeline,
    )


__all__ = [
    "add_intercept_column",
    "fit_numeric_imputer",
    "transform_numeric_imputer",
    "fit_correlated_feature_filter",
    "transform_correlated_feature_filter",
    "fit_standardizer",
    "transform_standardizer",
    "NumericPreprocessingPipeline",
    "PreparedDataSplit",
    "preprocess_train_valid_test",
    "prepare_model_ready_splits",
]
