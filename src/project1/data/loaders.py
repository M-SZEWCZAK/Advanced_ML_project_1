"""Dataset loading utilities for local ARFF files."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATASET_ARFF_PATHS = {
    "madelon": _PROJECT_ROOT / "data" / "raw" / "madelon" / "madelon.arff",
    "breast_cancer": _PROJECT_ROOT / "data" / "raw" / "breast_cancer" / "wdbc.arff",
    "spambase": _PROJECT_ROOT / "data" / "raw" / "spambase" / "spambase.arff",
    "ionosphere": _PROJECT_ROOT / "data" / "raw" / "ionosphere" / "ionosphere.arff",
}


def _decode_arff_value(value):
    """Decode ARFF values stored as bytes into regular Python strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _decode_arff_dataframe(df):
    """Decode byte-valued object columns in an ARFF-backed DataFrame."""
    decoded_df = df.copy()
    decoded_df.columns = [_decode_arff_value(column) for column in decoded_df.columns]

    for column in decoded_df.select_dtypes(include=["object"]).columns:
        decoded_df[column] = decoded_df[column].map(_decode_arff_value)

    return decoded_df


def _read_arff_to_dataframe(path):
    """Read a local ARFF file into a pandas DataFrame."""
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    return _decode_arff_dataframe(df)


def _prepare_features_and_binary_target(df, target_column, target_mapping):
    """Split a DataFrame into numeric features and a binary target vector."""
    feature_df = df.drop(columns=[target_column])
    target_raw = df[target_column].astype(str)

    unexpected_target_values = set(target_raw.unique()) - set(target_mapping)
    if unexpected_target_values:
        raise ValueError(f"Unexpected target values in dataset: {sorted(unexpected_target_values)}")

    X = feature_df.to_numpy(dtype=float)
    y = target_raw.map(target_mapping).to_numpy(dtype=int)
    feature_names = feature_df.columns.tolist()
    return X, y, feature_names


def _load_madelon_dataset():
    """Load the local Madelon ARFF file with `Class` mapped from `1/2` to `0/1`."""
    df = _read_arff_to_dataframe(_DATASET_ARFF_PATHS["madelon"])
    return _prepare_features_and_binary_target(df, target_column="Class", target_mapping={"1": 0, "2": 1})


def _load_breast_cancer_dataset():
    """Load the local breast cancer ARFF file with `Class` mapped from `1/2` to `0/1`."""
    df = _read_arff_to_dataframe(_DATASET_ARFF_PATHS["breast_cancer"])
    return _prepare_features_and_binary_target(df, target_column="Class", target_mapping={"1": 0, "2": 1})


def _load_spambase_dataset():
    """Load the local Spambase ARFF file with `class` already encoded as binary `0/1`."""
    df = _read_arff_to_dataframe(_DATASET_ARFF_PATHS["spambase"])
    return _prepare_features_and_binary_target(df, target_column="class", target_mapping={"0": 0, "1": 1})


def _load_ionosphere_dataset():
    """Load the local Ionosphere ARFF file with `class` mapped from `b/g` to `0/1`."""
    df = _read_arff_to_dataframe(_DATASET_ARFF_PATHS["ionosphere"])
    return _prepare_features_and_binary_target(df, target_column="class", target_mapping={"b": 0, "g": 1})


_DATASET_LOADERS = {
    "madelon": _load_madelon_dataset,
    "breast_cancer": _load_breast_cancer_dataset,
    "spambase": _load_spambase_dataset,
    "ionosphere": _load_ionosphere_dataset,
}


def get_supported_datasets():
    """Return the sorted list of dataset names supported by the local ARFF loaders."""
    return sorted(_DATASET_LOADERS)


def get_dataset_path(dataset_name):
    """Return the local ARFF path for a supported dataset name."""
    if dataset_name not in _DATASET_ARFF_PATHS:
        supported_datasets = ", ".join(get_supported_datasets())
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {supported_datasets}")
    return _DATASET_ARFF_PATHS[dataset_name]


def load_dataset(dataset_name):
    """Load a supported dataset by name and return `(X, y, feature_names)`."""
    if dataset_name in _DATASET_LOADERS:
        return _DATASET_LOADERS[dataset_name]()

    supported_datasets = ", ".join(get_supported_datasets())
    raise ValueError(
        f"Unsupported dataset: {dataset_name}. Supported datasets in this loader stage: {supported_datasets}"
    )


__all__ = ["get_supported_datasets", "get_dataset_path", "load_dataset"]
