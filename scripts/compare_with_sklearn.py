"""Compare the custom FISTA logistic lasso with sklearn LogisticRegression L1."""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from project1.data.loaders import get_supported_datasets, load_dataset
from project1.data.preprocessing import prepare_model_ready_splits
from project1.data.split import make_data_split_bundle
from project1.models.fista_logistic_lasso import FistaLogisticLassoRegressionClassifierFamily


def parse_args():
    """Parse command-line arguments for the comparison script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="breast_cancer",
        choices=get_supported_datasets(),
        help="Dataset used for the FISTA versus sklearn comparison.",
    )
    parser.add_argument(
        "--measure",
        default="auc",
        choices=["accuracy", "balanced_accuracy", "f1", "auc"],
        help="Validation metric used to choose the best regularization strength.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the data split and sklearn model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset assigned to the test split.",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset assigned to the validation split.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Absolute correlation threshold used during preprocessing.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=200,
        help="Number of FISTA iterations per lambda.",
    )
    return parser.parse_args()


def _add_intercept_column(X):
    """Return a copy of `X` with a leading intercept column of ones."""
    return np.column_stack((np.ones(X.shape[0]), X))


def _score_predictions(y_true, y_pred, y_proba):
    """Compute the comparison metrics for binary classification."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def _fit_best_sklearn_l1_logreg(X_train, y_train, X_valid, y_valid, lambdas, random_state, measure):
    """Fit sklearn L1 logistic regression over a grid and keep the best validation model."""
    best_model = None
    best_lambda = None
    best_score = -np.inf

    for lambda_ in lambdas:
        c_value = 1.0 / lambda_
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*Inconsistent values: penalty=l1.*", category=UserWarning)
            model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=c_value,
                random_state=random_state,
                max_iter=5000,
            )
            model.fit(X_train, y_train)
        valid_proba = model.predict_proba(X_valid)[:, 1]
        valid_pred = model.predict(X_valid)
        valid_scores = _score_predictions(y_valid, valid_pred, valid_proba)

        score_key = "roc_auc" if measure == "auc" else measure
        if valid_scores[score_key] > best_score:
            best_score = valid_scores[score_key]
            best_lambda = lambda_
            best_model = model

    return best_model, best_lambda, best_score


def run_comparison(
    dataset="breast_cancer",
    measure="auc",
    random_state=42,
    test_size=0.2,
    valid_size=0.2,
    correlation_threshold=0.95,
    n_iter=200,
):
    """Run the full data pipeline and compare FISTA with sklearn L1 logistic regression."""
    X, y, feature_names = load_dataset(dataset)
    split = make_data_split_bundle(
        X,
        y,
        random_state=random_state,
        test_size=test_size,
        valid_size=valid_size,
    )
    prepared = prepare_model_ready_splits(
        split.X_train,
        split.y_train,
        X_valid=split.X_valid,
        y_valid=split.y_valid,
        X_test=split.X_test,
        y_test=split.y_test,
        feature_names=feature_names,
        correlation_threshold=correlation_threshold,
    )

    X_train_fista = _add_intercept_column(prepared.X_train)
    X_valid_fista = _add_intercept_column(prepared.X_valid)
    X_test_fista = _add_intercept_column(prepared.X_test)

    lambdas = np.logspace(-3, 1, 10)

    fista_model = FistaLogisticLassoRegressionClassifierFamily(lambdas=lambdas, n_iter=n_iter)
    fista_model.fit(
        X_train_fista,
        prepared.y_train,
        X_validate=X_valid_fista,
        y_validate=prepared.y_valid,
        measure=measure,
    )
    fista_proba = fista_model.predict_proba(X_test_fista)[:, 1]
    fista_pred = fista_model.predict(X_test_fista)
    fista_scores = _score_predictions(prepared.y_test, fista_pred, fista_proba)

    sklearn_model, sklearn_best_lambda, sklearn_best_score = _fit_best_sklearn_l1_logreg(
        prepared.X_train,
        prepared.y_train,
        prepared.X_valid,
        prepared.y_valid,
        lambdas=lambdas,
        random_state=random_state,
        measure=measure,
    )
    sklearn_proba = sklearn_model.predict_proba(prepared.X_test)[:, 1]
    sklearn_pred = sklearn_model.predict(prepared.X_test)
    sklearn_scores = _score_predictions(prepared.y_test, sklearn_pred, sklearn_proba)

    return {
        "dataset": dataset,
        "validation_measure": measure,
        "n_features_before_preprocessing": len(feature_names),
        "n_features_after_preprocessing": len(prepared.feature_names),
        "fista": {
            "best_lambda": float(fista_model.best_lambda),
            "best_validation_score": None if fista_model.best_score_ is None else float(fista_model.best_score_),
            **{metric: float(value) for metric, value in fista_scores.items()},
        },
        "sklearn_l1": {
            "best_lambda": float(sklearn_best_lambda),
            "best_validation_score": float(sklearn_best_score),
            **{metric: float(value) for metric, value in sklearn_scores.items()},
        },
    }


def _format_metrics(metrics):
    """Format a metric dictionary into readable terminal lines."""
    ordered_keys = ["best_lambda", "best_validation_score", "accuracy", "balanced_accuracy", "f1", "roc_auc"]
    return "\n".join(f"  {key}: {metrics[key]}" for key in ordered_keys)


def main():
    """Run the comparison and print a compact summary to the terminal."""
    args = parse_args()
    result = run_comparison(
        dataset=args.dataset,
        measure=args.measure,
        random_state=args.random_state,
        test_size=args.test_size,
        valid_size=args.valid_size,
        correlation_threshold=args.correlation_threshold,
        n_iter=args.n_iter,
    )

    print("FISTA vs sklearn L1 logistic regression")
    print(f"dataset: {result['dataset']}")
    print(f"validation_measure: {result['validation_measure']}")
    print(f"n_features_before_preprocessing: {result['n_features_before_preprocessing']}")
    print(f"n_features_after_preprocessing: {result['n_features_after_preprocessing']}")
    print("fista:")
    print(_format_metrics(result["fista"]))
    print("sklearn_l1:")
    print(_format_metrics(result["sklearn_l1"]))


if __name__ == "__main__":
    main()
