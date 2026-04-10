"""Missing-label generation schemes: MCAR, MAR1, MAR2, MNAR."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def logistic_missingness_loss(xc, bias, steepness, missing_proba):
    """Measure how close the average logistic missingness is to a target rate."""
    proba = expit(steepness * (xc - bias))
    return (proba.mean() - missing_proba) ** 2


def MCAR(X, Y, missing_proba, seed=None):
    """Generate labels missing completely at random with probability `missing_proba`."""
    if seed is not None:
        np.random.seed(seed)

    length = len(Y)
    new_y = np.zeros(length)
    make_missing = np.random.uniform(0, 1, length)

    for j in range(length):
        if make_missing[j] < missing_proba:
            new_y[j] = -1
        else:
            new_y[j] = Y[j]

    return new_y


def MAR1(X, Y, missing_proba, missing_influence_col_index, steepness=1.0, seed=None):
    """Generate labels missing at random based on a single standardized feature."""
    if missing_influence_col_index < 0 or missing_influence_col_index >= X.shape[1]:
        raise ValueError(f"Wrong column index. Index should be between 0 and {X.shape[1] - 1}")

    if seed is not None:
        np.random.seed(seed)

    length = len(Y)
    new_y = np.zeros(length)
    xc = X[:, missing_influence_col_index]
    result = minimize_scalar(
        lambda bias: logistic_missingness_loss(xc, bias, steepness, missing_proba),
        bounds=(-10, 10),
        method="bounded",
    )
    bias = result.x
    per_row_probabilities = expit(steepness * (xc - bias))
    make_missing = np.random.uniform(0, 1, length)

    for j in range(length):
        if make_missing[j] < per_row_probabilities[j]:
            new_y[j] = -1
        else:
            new_y[j] = Y[j]

    return new_y


def MAR2(X, Y, missing_proba, steepness=1.0, weights=None, seed=None):
    """Generate labels missing at random based on a weighted combination of features."""
    if seed is not None:
        np.random.seed(seed)

    n_cols = X.shape[1]
    length = len(Y)
    if weights is None:
        weights_used = np.ones(n_cols) / np.sqrt(n_cols)
    else:
        weights_used = np.array(weights) / np.linalg.norm(weights)

    xc = X @ weights_used
    result = minimize_scalar(
        lambda bias: logistic_missingness_loss(xc, bias, steepness, missing_proba),
        bounds=(-10, 10),
        method="bounded",
    )
    bias = result.x
    per_row_probabilities = expit(steepness * (xc - bias))
    make_missing = np.random.uniform(0, 1, length)
    new_y = np.zeros(length)

    for j in range(length):
        if make_missing[j] < per_row_probabilities[j]:
            new_y[j] = -1
        else:
            new_y[j] = Y[j]

    return new_y


def MNAR(X, Y, missing_proba, seed=None):
    """Generate labels missing not at random using label-aware probabilities."""
    if seed is not None:
        np.random.seed(seed)

    length = len(Y)
    new_y = np.zeros(length)
    lr = LogisticRegression()
    lr.fit(X, Y)
    probabilities = lr.predict_proba(X)[:, 1]
    unnormalized_missing_probas = 0.5 * probabilities + 0.5 * Y
    mean_missing_proba = np.mean(unnormalized_missing_probas)
    normalized_missing_probas = unnormalized_missing_probas * missing_proba / mean_missing_proba.astype(float)
    make_missing = np.random.uniform(0, 1, length)

    for j in range(length):
        if normalized_missing_probas[j] > 1:
            normalized_missing_probas[j] = 1
        if make_missing[j] < normalized_missing_probas[j]:
            new_y[j] = -1
        else:
            new_y[j] = Y[j]

    return new_y


__all__ = ["logistic_missingness_loss", "MCAR", "MAR1", "MAR2", "MNAR"]
