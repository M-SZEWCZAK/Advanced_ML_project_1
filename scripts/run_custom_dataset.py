"""Run a single experiment on a user-provided CSV dataset with a binary target column."""

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from project1.data.preprocessing import prepare_model_ready_splits
from project1.data.split import make_data_split
from project1.experiments.runner import (
    _build_model,
    _compute_test_metrics,
    _generate_missing_train_labels,
    _to_probability_scores,
)


def parse_args():
    """Parse command-line arguments for running one experiment on a custom CSV dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", required=True, help="Path to the input CSV file.")
    parser.add_argument("--target-column", required=True, help="Name of the binary target column in the CSV file.")
    parser.add_argument(
        "--scheme",
        required=True,
        choices=["mcar", "mar1", "mar2", "mnar"],
        help="Missing-label mechanism applied only to the training labels.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["naive", "oracle", "unlabeled"],
        help="Modeling approach used in the experiment.",
    )
    parser.add_argument("--missing-rate", required=True, type=float, help="Missing-label rate used in training only.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed for splitting and model fitting.")
    parser.add_argument(
        "--label-completion-method",
        default="logistic",
        choices=["logistic", "knn", "prior"],
        help="Label-completion method used when --method unlabeled.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction assigned to the test split.")
    parser.add_argument("--valid-size", type=float, default=0.2, help="Fraction assigned to the validation split.")
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Absolute correlation threshold for dropping strongly correlated features.",
    )
    return parser.parse_args()


def _load_custom_csv(csv_path, target_column):
    """Load a custom CSV file and return `(X, y, feature_names, target_mapping)`."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV file.")

    feature_df = df.drop(columns=[target_column]).copy()
    target_series = df[target_column].copy()

    if feature_df.empty:
        raise ValueError("The CSV file must contain at least one feature column besides the target.")

    non_numeric_columns = [column for column in feature_df.columns if not pd.api.types.is_numeric_dtype(feature_df[column])]
    if non_numeric_columns:
        raise ValueError(
            "All feature columns must be numeric. Non-numeric columns: " + ", ".join(non_numeric_columns)
        )

    y, target_mapping = _prepare_binary_target(target_series)
    X = feature_df.to_numpy(dtype=float)
    feature_names = feature_df.columns.tolist()
    return X, y, feature_names, target_mapping


def _prepare_binary_target(target_series):
    """Map a binary target column to a one-dimensional NumPy array with values 0 and 1."""
    if target_series.isna().any():
        raise ValueError("Target column must not contain missing values.")

    unique_values = list(pd.unique(target_series))
    if len(unique_values) != 2:
        raise ValueError("Target column must contain exactly two unique classes for binary classification.")

    if set(unique_values) == {0, 1}:
        mapping = {0: 0, 1: 1}
    elif set(unique_values) == {"0", "1"}:
        mapping = {"0": 0, "1": 1}
    else:
        sorted_values = sorted(unique_values, key=lambda value: str(value))
        mapping = {sorted_values[0]: 0, sorted_values[1]: 1}

    y = target_series.map(mapping)
    if y.isna().any():
        raise ValueError("Failed to map the target column to binary values 0/1.")

    return y.to_numpy(dtype=int), {str(key): value for key, value in mapping.items()}


def _format_result(result, csv_path, target_column, target_mapping):
    """Format the experiment result into a readable terminal summary."""
    ordered_keys = [
        "scheme",
        "method",
        "label_completion_method",
        "seed",
        "missing_rate",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "train_size",
        "valid_size",
        "test_size",
        "n_features_before_preprocessing",
        "n_features_after_preprocessing",
        "observed_label_fraction_train",
    ]

    lines = [
        "Custom dataset experiment result",
        f"csv_path: {csv_path}",
        f"target_column: {target_column}",
        f"target_mapping: {target_mapping}",
    ]
    for key in ordered_keys:
        if key in result:
            lines.append(f"{key}: {result[key]}")
    return "\n".join(lines)


def main():
    """Run the full experiment pipeline for a user-provided CSV dataset."""
    args = parse_args()

    X, y, feature_names, target_mapping = _load_custom_csv(args.csv_path, args.target_column)
    X_train, X_valid, X_test, y_train, y_valid, y_test = make_data_split(
        X,
        y,
        random_state=args.seed,
        test_size=args.test_size,
        valid_size=args.valid_size,
    )
    prepared = prepare_model_ready_splits(
        X_train,
        y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        correlation_threshold=args.correlation_threshold,
    )

    y_train_obs = _generate_missing_train_labels(
        prepared.X_train,
        prepared.y_train,
        missingness=args.scheme,
        missing_proba=args.missing_rate,
        random_state=args.seed,
    )

    model = _build_model(
        args.method,
        random_state=args.seed,
        label_completion_method=args.label_completion_method,
    )
    if args.method == "naive":
        model.fit(prepared.X_train, y_train_obs)
    elif args.method == "oracle":
        model.fit(prepared.X_train, prepared.y_train)
    else:
        model.fit(
            prepared.X_train,
            y_train_obs,
            X_validate=prepared.X_valid,
            y_validate=prepared.y_valid,
        )

    y_test_pred = model.predict(prepared.X_test)
    y_test_score = _to_probability_scores(model, prepared.X_test)
    metrics = _compute_test_metrics(prepared.y_test, y_test_pred, y_test_score)

    result = {
        "scheme": args.scheme,
        "method": args.method,
        "label_completion_method": args.label_completion_method if args.method == "unlabeled" else None,
        "seed": args.seed,
        "missing_rate": args.missing_rate,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "train_size": int(prepared.X_train.shape[0]),
        "valid_size": int(prepared.X_valid.shape[0]),
        "test_size": int(prepared.X_test.shape[0]),
        "n_features_before_preprocessing": int(len(feature_names)),
        "n_features_after_preprocessing": int(len(prepared.feature_names)),
        "observed_label_fraction_train": float((y_train_obs != -1).mean()),
    }

    print(_format_result(result, args.csv_path, args.target_column, target_mapping))


if __name__ == "__main__":
    main()
