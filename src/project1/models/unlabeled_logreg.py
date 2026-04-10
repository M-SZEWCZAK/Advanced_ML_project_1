"""Unlabeled logistic regression methods based on label completion."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from project1.models.fista_logistic_lasso import FistaLogisticLassoRegressionClassifierFamily


class UnlabeledLogReg:
    """Complete missing binary labels and fit a downstream logistic-lasso classifier."""

    def __init__(
        self,
        imputation_type,
        threshold=0.5,
        random_state=2137,
        lambdas=None,
        n_iter=100,
        neighbor_count=5,
    ):
        if imputation_type not in {"logistic", "knn", "prior"}:
            raise ValueError("Invalid imputation type: expected one of logistic, knn, prior")

        self.imputation_type = imputation_type
        self.threshold = threshold
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_iter = n_iter
        self.neighbor_count = neighbor_count

        self.completed_labels_ = None
        self.classifier_ = None

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
    def _to_numpy_labels(Y):
        """Convert label input to a one-dimensional NumPy array."""
        if hasattr(Y, "to_numpy"):
            y_array = Y.to_numpy()
        else:
            y_array = np.asarray(Y)

        if y_array.ndim != 1:
            y_array = np.ravel(y_array)
        return y_array.astype(float, copy=False)

    def _split_observed_and_missing(self, X, Y):
        """Split features and labels into observed and missing-label subsets."""
        X_array = self._to_numpy_features(X)
        y_array = self._to_numpy_labels(Y).copy()

        if len(X_array) != len(y_array):
            raise ValueError("X and Y must contain the same number of rows.")

        observed_mask = y_array != -1
        if not np.any(observed_mask):
            raise ValueError("At least one observed label is required for label completion.")

        return X_array, y_array, observed_mask

    @staticmethod
    def _fill_with_single_class(y, missing_mask, observed_labels):
        """Fill missing labels when the observed subset contains only one class."""
        y[missing_mask] = observed_labels[0]
        return y

    def _logistic_imputation(self, X, Y):
        """Impute missing labels with a logistic regression fitted on observed labels."""
        X_array, y_array, observed_mask = self._split_observed_and_missing(X, Y)
        missing_mask = ~observed_mask

        if not np.any(missing_mask):
            return y_array.astype(int)

        observed_labels = y_array[observed_mask].astype(int)
        unique_labels = np.unique(observed_labels)
        if len(unique_labels) == 1:
            return self._fill_with_single_class(y_array, missing_mask, unique_labels).astype(int)

        model = LogisticRegression(random_state=self.random_state)
        model.fit(X_array[observed_mask], observed_labels)
        y_array[missing_mask] = model.predict(X_array[missing_mask])
        return y_array.astype(int)

    def _knn_imputation(self, X, Y):
        """Impute missing labels with k-nearest neighbours fitted on observed labels."""
        X_array, y_array, observed_mask = self._split_observed_and_missing(X, Y)
        missing_mask = ~observed_mask

        if not np.any(missing_mask):
            return y_array.astype(int)

        observed_labels = y_array[observed_mask].astype(int)
        knn = KNeighborsClassifier(n_neighbors=self.neighbor_count)
        knn.fit(X_array[observed_mask], observed_labels)
        y_array[missing_mask] = knn.predict(X_array[missing_mask])
        return y_array.astype(int)

    def _prior_probability_imputation(self, Y):
        """Impute missing labels from the observed positive-class prior."""
        y_array = self._to_numpy_labels(Y).copy()
        observed_mask = y_array != -1
        missing_mask = ~observed_mask

        if not np.any(observed_mask):
            raise ValueError("At least one observed label is required for label completion.")
        if not np.any(missing_mask):
            return y_array.astype(int)

        observed_labels = y_array[observed_mask]
        positive_prior = np.mean(observed_labels)
        rng = np.random.default_rng(self.random_state)
        y_array[missing_mask] = rng.binomial(1, positive_prior, np.sum(missing_mask))
        return y_array.astype(int)

    def complete_labels(self, X, Y):
        """Return a completed label vector using the configured imputation strategy."""
        if self.imputation_type == "logistic":
            completed_labels = self._logistic_imputation(X, Y)
        elif self.imputation_type == "knn":
            completed_labels = self._knn_imputation(X, Y)
        else:
            completed_labels = self._prior_probability_imputation(Y)

        self.completed_labels_ = completed_labels
        return completed_labels

    def fit(self, X, Y, X_validate=None, y_validate=None, measure="accuracy"):
        """Complete missing labels in the training set and fit the downstream classifier."""
        X_train = self._to_numpy_features(X)
        new_y = self.complete_labels(X_train, Y)

        X_val_array = None if X_validate is None else self._to_numpy_features(X_validate)
        y_val_array = None if y_validate is None else self._to_numpy_labels(y_validate).astype(int)

        classifier = FistaLogisticLassoRegressionClassifierFamily(
            threshold=self.threshold,
            random_state=self.random_state,
            lambdas=self.lambdas,
            n_iter=self.n_iter,
        )
        classifier.fit(X_train, new_y, X_val_array, y_val_array, measure=measure)

        self.classifier_ = classifier
        return self

    def predict_proba(self, X):
        """Predict class probabilities with the fitted downstream classifier."""
        if self.classifier_ is None:
            raise ValueError("Model is not fitted yet. Call fit before predict_proba.")
        return self.classifier_.predict_proba(self._to_numpy_features(X))

    def predict(self, X):
        """Predict class labels with the fitted downstream classifier."""
        if self.classifier_ is None:
            raise ValueError("Model is not fitted yet. Call fit before predict.")
        return self.classifier_.predict(self._to_numpy_features(X))


__all__ = ["UnlabeledLogReg"]
