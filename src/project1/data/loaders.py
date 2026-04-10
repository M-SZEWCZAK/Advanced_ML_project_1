"""Dataset loading utilities for local ARFF files."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATASET_ARFF_PATHS = {
    "breast_cancer": _PROJECT_ROOT / "data" / "raw" / "breast_cancer" / "wdbc.arff",
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


def _load_breast_cancer_dataset():
    """Load the local breast cancer ARFF file and return features, labels, and feature names."""
    df = _read_arff_to_dataframe(_DATASET_ARFF_PATHS["breast_cancer"])

    target_column = "Class"
    feature_df = df.drop(columns=[target_column])
    target_raw = df[target_column].astype(str)

    target_mapping = {"1": 0, "2": 1}
    if set(target_raw.unique()) - set(target_mapping):
        raise ValueError("Unexpected target values in breast_cancer ARFF file.")

    X = feature_df.to_numpy(dtype=float)
    y = target_raw.map(target_mapping).to_numpy(dtype=int)
    feature_names = feature_df.columns.tolist()

    return X, y, feature_names


def load_dataset(dataset_name):
    """Load a supported dataset by name and return `(X, y, feature_names)`."""
    if dataset_name == "breast_cancer":
        return _load_breast_cancer_dataset()

    supported_datasets = ", ".join(sorted(_DATASET_ARFF_PATHS))
    raise ValueError(
        f"Unsupported dataset: {dataset_name}. Supported datasets in this loader stage: {supported_datasets}"
    )
