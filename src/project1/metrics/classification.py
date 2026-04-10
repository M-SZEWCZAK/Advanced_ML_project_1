"""Classification metrics used in validation and experiments."""

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


_VALIDATION_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "balanced_accuracy",
    "average_precision",
    "spc",
}


def score_binary_classification(y_true, y_pred, y_score, measure="accuracy"):
    """Score binary classification outputs with a metric used by the FISTA model."""
    if measure == "accuracy":
        return accuracy_score(y_true, y_pred)
    if measure == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    if measure == "recall":
        return recall_score(y_true, y_pred, zero_division=0)
    if measure == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    if measure == "auc":
        return roc_auc_score(y_true, y_score)
    if measure == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if measure in {"average_precision", "spc"}:
        return average_precision_score(y_true, y_score)

    valid_measures = ", ".join(sorted(_VALIDATION_METRICS))
    raise ValueError(f"Invalid measure: {measure}. Valid measures are: {valid_measures}")


__all__ = ["score_binary_classification"]
