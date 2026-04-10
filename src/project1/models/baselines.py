"""Baseline logistic-regression models for Project 1."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class _BaseLogReg:
    """Shared helpers for baseline logistic-regression models."""

    def __init__(self, random_state=42, **logreg_kwargs):
        self.random_state = random_state
        self.logreg_kwargs = logreg_kwargs
        self.model_ = None
        self.is_constant_predictor_ = False
        self.constant_class_ = None

    @staticmethod
    def _to_numpy_features(X):
        """Convert feature input to a two-dimensional NumPy array."""
        if hasattr(X, "to_numpy"):
            X_array = X.to_numpy()
        else:
            X_array = np.asarray(X)

        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array or DataFrame.")
        return X_array

    @staticmethod
    def _to_numpy_labels(y):
        """Convert label input to a one-dimensional NumPy array."""
        if hasattr(y, "to_numpy"):
            y_array = y.to_numpy()
        else:
            y_array = np.asarray(y)
        return np.ravel(y_array)

    def _make_model(self):
        """Create the sklearn logistic-regression backend."""
        return LogisticRegression(random_state=self.random_state, **self.logreg_kwargs)

    def _ensure_is_fitted(self):
        """Ensure that the sklearn backend has already been fitted."""
        if self.model_ is None and not self.is_constant_predictor_:
            raise ValueError("Model is not fitted yet. Call fit before prediction.")

    def predict(self, X):
        """Predict binary labels for the provided feature matrix."""
        self._ensure_is_fitted()
        X_array = self._to_numpy_features(X)
        if self.is_constant_predictor_:
            return np.full(X_array.shape[0], self.constant_class_, dtype=int)
        return self.model_.predict(X_array)

    def predict_proba(self, X):
        """Predict class probabilities for the provided feature matrix."""
        self._ensure_is_fitted()
        X_array = self._to_numpy_features(X)
        if self.is_constant_predictor_:
            positive_class_proba = float(self.constant_class_)
            negative_class_proba = 1.0 - positive_class_proba
            return np.tile([negative_class_proba, positive_class_proba], (X_array.shape[0], 1))
        return self.model_.predict_proba(X_array)


class NaiveLogReg(_BaseLogReg):
    """Baseline logistic regression trained only on samples with observed labels."""

    def fit(self, X, y_obs):
        """Fit the model on rows for which the observed label is not equal to `-1`."""
        X_array = self._to_numpy_features(X)
        y_array = self._to_numpy_labels(y_obs)

        if len(X_array) != len(y_array):
            raise ValueError("X and y_obs must contain the same number of rows.")

        observed_mask = y_array != -1
        if not np.any(observed_mask):
            raise ValueError("NaiveLogReg requires at least one observed label.")

        X_observed = X_array[observed_mask]
        y_observed = y_array[observed_mask].astype(int)
        unique_classes = np.unique(y_observed)

        self.model_ = None
        self.is_constant_predictor_ = False
        self.constant_class_ = None

        if len(unique_classes) < 2:
            # Fallback for degenerate observed-label subsets: predict the only available class.
            self.is_constant_predictor_ = True
            self.constant_class_ = int(unique_classes[0])
            return self

        self.model_ = self._make_model()
        self.model_.fit(X_observed, y_observed)
        return self


class OracleLogReg(_BaseLogReg):
    """Baseline logistic regression trained on the complete ground-truth labels."""

    def fit(self, X, y):
        """Fit the model on the full label vector without missing labels."""
        X_array = self._to_numpy_features(X)
        y_array = self._to_numpy_labels(y).astype(int)

        if len(X_array) != len(y_array):
            raise ValueError("X and y must contain the same number of rows.")
        if np.any(y_array == -1):
            raise ValueError("OracleLogReg expects fully observed labels without -1 values.")
        if len(np.unique(y_array)) < 2:
            raise ValueError("OracleLogReg requires at least two classes to fit LogisticRegression.")

        self.model_ = self._make_model()
        self.model_.fit(X_array, y_array)
        return self


__all__ = ["NaiveLogReg", "OracleLogReg"]
