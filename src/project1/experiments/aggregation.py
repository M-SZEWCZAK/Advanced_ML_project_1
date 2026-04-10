"""Helpers for collecting and aggregating experiment results."""

from pathlib import Path

import pandas as pd


DEFAULT_RESULT_COLUMNS = [
    "dataset",
    "scheme",
    "method",
    "seed",
    "missing_rate",
    "status",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "roc_auc",
    "error_type",
    "error_message",
]

DEFAULT_METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "roc_auc",
]


def results_to_dataframe(results, columns=None):
    """Convert a list of experiment-result dictionaries into a pandas DataFrame."""
    result_df = pd.DataFrame(results)

    if columns is None:
        ordered_columns = [column for column in DEFAULT_RESULT_COLUMNS if column in result_df.columns]
        remaining_columns = [column for column in result_df.columns if column not in ordered_columns]
        return result_df.loc[:, ordered_columns + remaining_columns]

    return result_df.loc[:, list(columns)]


def save_results_to_csv(results, output_path, columns=None, index=False):
    """Save experiment results to a CSV file and return the DataFrame that was written."""
    result_df = results_to_dataframe(results, columns=columns)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=index)
    return result_df


def aggregate_results(results, groupby_columns=None, metric_columns=None):
    """Aggregate successful experiment results by groups using mean and standard deviation for metrics."""
    result_df = results_to_dataframe(results)

    if groupby_columns is None:
        groupby_columns = ["dataset", "scheme", "method", "missing_rate"]
    if metric_columns is None:
        metric_columns = DEFAULT_METRIC_COLUMNS

    if "status" in result_df.columns:
        result_df = result_df[result_df["status"] != "error"].copy()

    for column in metric_columns:
        if column not in result_df.columns:
            result_df[column] = pd.NA

    if result_df.empty:
        metric_stat_columns = [f"{metric}_{stat}" for metric in metric_columns for stat in ("mean", "std")]
        return pd.DataFrame(columns=[*groupby_columns, *metric_stat_columns])

    missing_columns = [column for column in groupby_columns if column not in result_df.columns]
    if missing_columns:
        raise ValueError(f"Missing groupby columns in results: {missing_columns}")

    aggregated_df = (
        result_df.groupby(groupby_columns, dropna=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated_df.columns = [
        "_".join(str(part) for part in column if part).rstrip("_") for column in aggregated_df.columns.to_flat_index()
    ]
    return aggregated_df


__all__ = [
    "DEFAULT_RESULT_COLUMNS",
    "DEFAULT_METRIC_COLUMNS",
    "results_to_dataframe",
    "save_results_to_csv",
    "aggregate_results",
]
