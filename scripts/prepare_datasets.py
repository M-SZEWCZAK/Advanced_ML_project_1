"""Prepare a local ARFF dataset into train/valid/test arrays ready for modeling."""

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from project1.data.loaders import get_dataset_path, get_supported_datasets, load_dataset
from project1.data.preprocessing import prepare_model_ready_splits
from project1.data.split import make_data_split_bundle


def _format_class_distribution(y):
    """Format class counts for concise console output."""
    classes, counts = np.unique(y, return_counts=True)
    return ", ".join(f"class {int(label)}: {int(count)}" for label, count in zip(classes, counts))


def parse_args():
    """Parse command-line arguments for dataset preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name", choices=get_supported_datasets(), help="Dataset name to prepare.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used for splitting.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction assigned to the test split.")
    parser.add_argument("--valid-size", type=float, default=0.2, help="Fraction assigned to the validation split.")
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Absolute correlation threshold for dropping redundant features.",
    )
    return parser.parse_args()


def main():
    """Run the end-to-end local data preparation pipeline."""
    args = parse_args()

    X, y, feature_names = load_dataset(args.dataset_name)
    split = make_data_split_bundle(
        X,
        y,
        random_state=args.random_state,
        test_size=args.test_size,
        valid_size=args.valid_size,
    )
    prepared = prepare_model_ready_splits(
        split.X_train,
        split.y_train,
        X_valid=split.X_valid,
        y_valid=split.y_valid,
        X_test=split.X_test,
        y_test=split.y_test,
        feature_names=feature_names,
        correlation_threshold=args.correlation_threshold,
    )

    print(f"Dataset: {args.dataset_name}")
    print(f"ARFF path: {get_dataset_path(args.dataset_name)}")
    print(f"Number of observations: {len(y)}")
    print(f"Number of features before preprocessing: {len(feature_names)}")
    print(f"Number of features after preprocessing: {len(prepared.feature_names)}")
    print(
        "Split sizes: "
        f"train={prepared.X_train.shape[0]}, valid={prepared.X_valid.shape[0]}, test={prepared.X_test.shape[0]}"
    )
    print(
        "Processed array shapes: "
        f"train={prepared.X_train.shape}, valid={prepared.X_valid.shape}, test={prepared.X_test.shape}"
    )
    print(f"Train labels: {_format_class_distribution(prepared.y_train)}")
    print(f"Valid labels: {_format_class_distribution(prepared.y_valid)}")
    print(f"Test labels: {_format_class_distribution(prepared.y_test)}")
    print(f"First final features: {prepared.feature_names[:10]}")


if __name__ == "__main__":
    main()
