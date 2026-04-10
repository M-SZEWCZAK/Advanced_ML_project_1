"""Data-splitting utilities for Project 1."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


def _to_numpy_features(X):
    """Convert feature input to a NumPy array while preserving two-dimensional shape."""
    if hasattr(X, "to_numpy"):
        X_array = X.to_numpy()
    else:
        X_array = np.asarray(X)

    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array or DataFrame.")
    return X_array


def _to_numpy_labels(y):
    """Convert label input to a one-dimensional NumPy array."""
    if hasattr(y, "to_numpy"):
        y_array = y.to_numpy()
    else:
        y_array = np.asarray(y)
    return np.ravel(y_array)


def _can_stratify(y):
    """Return whether stratified splitting is safe for the provided labels."""
    unique_labels, counts = np.unique(y, return_counts=True)
    return len(unique_labels) > 1 and np.all(counts >= 2)


@dataclass
class DataSplit:
    """Container for train, validation, and test splits."""

    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray


def make_data_split(X, y, random_state=42, test_size=0.2, valid_size=0.2):
    """Split data into train, validation, and test subsets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Label vector.
    random_state : int, default=42
        Random seed used for reproducible splits.
    test_size : float, default=0.2
        Fraction of the full dataset assigned to the test split.
    valid_size : float, default=0.2
        Fraction of the full dataset assigned to the validation split.

    Returns
    -------
    tuple
        `(X_train, X_valid, X_test, y_train, y_valid, y_test)`.
    """
    X_array = _to_numpy_features(X)
    y_array = _to_numpy_labels(y)

    if len(X_array) != len(y_array):
        raise ValueError("X and y must contain the same number of samples.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < valid_size < 1:
        raise ValueError("valid_size must be between 0 and 1.")
    if test_size + valid_size >= 1:
        raise ValueError("test_size + valid_size must be smaller than 1.")

    stratify_full = y_array if _can_stratify(y_array) else None
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X_array,
        y_array,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_full,
    )

    valid_fraction_within_train_valid = valid_size / (1 - test_size)
    stratify_train_valid = y_train_valid if _can_stratify(y_train_valid) else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=valid_fraction_within_train_valid,
        random_state=random_state,
        stratify=stratify_train_valid,
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def make_data_split_bundle(X, y, random_state=42, test_size=0.2, valid_size=0.2):
    """Split data and return the result wrapped in a `DataSplit` container."""
    X_train, X_valid, X_test, y_train, y_valid, y_test = make_data_split(
        X,
        y,
        random_state=random_state,
        test_size=test_size,
        valid_size=valid_size,
    )
    return DataSplit(X_train=X_train, X_valid=X_valid, X_test=X_test, y_train=y_train, y_valid=y_valid, y_test=y_test)


__all__ = ["DataSplit", "make_data_split", "make_data_split_bundle"]
