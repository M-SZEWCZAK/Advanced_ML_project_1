"""Own implementation of logistic lasso regression with FISTA."""

import numpy as np

from project1.metrics.classification import score_binary_classification
from project1.utils.math_utils import sigmoid


class LassoAuxiliary:
    """Helper methods for optimizing logistic lasso with FISTA."""

    @staticmethod
    def lasso_penalty(w, llambda):
        """Compute the L1 penalty while leaving the intercept unpenalized."""
        return llambda * np.linalg.norm(w[1:], ord=1)

    @staticmethod
    def logistic_loss(w, X, y):
        """Compute the average logistic loss for binary targets."""
        probabilities = sigmoid(X @ w)
        return -np.mean(y * np.log(probabilities + 1e-12) + (1 - y) * np.log(1 - probabilities + 1e-12))

    @staticmethod
    def logistic_loss_gradient(w, X, y):
        """Compute the gradient of the average logistic loss."""
        probabilities = sigmoid(X @ w)
        return X.T @ (probabilities - y) / len(y)

    @staticmethod
    def lipschitzvalue(X):
        """Estimate the Lipschitz constant for logistic loss on design matrix `X`."""
        n_samples = X.shape[0]
        return (1 / (4 * n_samples)) * (np.linalg.norm(X, ord=2) ** 2)

    @staticmethod
    def soft_thresholding_operator_for_lasso(w, shrinkage):
        """Apply soft-thresholding to all coefficients except the intercept."""
        thresholded = w.copy()
        thresholded[1:] = np.sign(thresholded[1:]) * np.maximum(np.abs(thresholded[1:]) - shrinkage, 0)
        return thresholded

    @staticmethod
    def fista_lasso_solver(X, y, llambda, n_iter=100):
        """Fit logistic lasso coefficients with FISTA for a fixed `llambda`."""
        n_features = X.shape[1]
        lipschitz_constant = LassoAuxiliary.lipschitzvalue(X)
        shrinkage = llambda / lipschitz_constant

        x_curr = np.zeros(n_features)
        z_curr = np.zeros(n_features)
        t_prev = 1.0

        for _ in range(n_iter):
            gradient_step = z_curr - (1 / lipschitz_constant) * LassoAuxiliary.logistic_loss_gradient(z_curr, X, y)
            x_next = LassoAuxiliary.soft_thresholding_operator_for_lasso(gradient_step, shrinkage)
            t_next = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
            z_next = x_next + ((t_prev - 1) / t_next) * (x_next - x_curr)

            x_curr = x_next
            z_curr = z_next
            t_prev = t_next

        return x_curr


class FistaLogisticLassoRegressionClassifierFamily:
    """Train logistic-lasso models over a lambda grid and select the best one on validation data."""

    def __init__(self, threshold=0.5, random_state=2137, lambdas=None, n_iter=100):
        self.threshold = threshold
        self.random_state = random_state
        self.n_iter = n_iter
        self.lambdas = np.logspace(-3, 1, 10) if lambdas is None else np.asarray(lambdas, dtype=float)

        self.best_lambda = None
        self.best_score_ = None
        self.validation_measure_ = None
        self.weights = {key: None for key in self.lambdas}

    def _get_weights_for_lambda(self, lambda_=None):
        """Return coefficients for the requested lambda or the selected best lambda."""
        selected_lambda = self.best_lambda if lambda_ is None else lambda_
        if selected_lambda is None:
            raise ValueError("Model is not fitted yet. Call fit before prediction or validation.")
        if selected_lambda not in self.weights or self.weights[selected_lambda] is None:
            raise ValueError(f"No fitted weights found for lambda={selected_lambda}.")
        return self.weights[selected_lambda]

    def _predict_positive_proba(self, X, w):
        """Return positive-class probabilities for a design matrix and coefficient vector."""
        return sigmoid(X @ w)

    def validate(self, X_validate, y_validate, lambda_=None, measure="accuracy"):
        """Score the model on validation data using the selected metric."""
        if lambda_ is None:
            positive_class_proba = self.predict_proba(X_validate)[:, 1]
        else:
            w = self._get_weights_for_lambda(lambda_)
            positive_class_proba = self._predict_positive_proba(X_validate, w)
        predictions = (positive_class_proba >= self.threshold).astype(int)
        return score_binary_classification(y_validate, predictions, positive_class_proba, measure=measure)

    def fit(self, X_train, y_train, X_validate=None, y_validate=None, measure="accuracy"):
        """Fit one model per lambda and pick the best lambda on validation data."""
        self.validation_measure_ = measure

        if (X_validate is None) != (y_validate is None):
            raise ValueError("Provide both X_validate and y_validate, or neither.")

        if X_validate is None and len(self.lambdas) > 1:
            raise ValueError("Validation data is required to select the best lambda.")

        best_lambda = self.lambdas[0]
        best_result = -np.inf

        for lambda_ in self.lambdas:
            self.weights[lambda_] = LassoAuxiliary.fista_lasso_solver(X_train, y_train, lambda_, n_iter=self.n_iter)

            if X_validate is None:
                val_result = 0.0
            else:
                val_result = self.validate(X_validate, y_validate, lambda_=lambda_, measure=measure)

            if val_result > best_result:
                best_lambda = lambda_
                best_result = val_result

        self.best_lambda = best_lambda
        self.best_score_ = None if X_validate is None else best_result
        return self

    def predict_proba(self, X):
        """Return class probabilities for the best selected lambda."""
        w = self._get_weights_for_lambda()
        positive_class_proba = self._predict_positive_proba(X, w)
        return np.column_stack((1 - positive_class_proba, positive_class_proba))

    def predict(self, X):
        """Return binary predictions for the best selected lambda."""
        positive_class_proba = self.predict_proba(X)[:, 1]
        return (positive_class_proba >= self.threshold).astype(int)


__all__ = ["LassoAuxiliary", "FistaLogisticLassoRegressionClassifierFamily"]
