"""Run a single baseline or unlabeled-logreg experiment from the command line."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from project1.data.loaders import get_supported_datasets
from project1.experiments.runner import run_single_experiment


def parse_args():
    """Parse command-line arguments for a single experiment run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name", choices=get_supported_datasets(), help="Dataset used in the experiment.")
    parser.add_argument(
        "model_name",
        choices=["naive_logreg", "oracle_logreg", "unlabeled_logreg"],
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--missingness",
        default="MCAR",
        choices=["MCAR", "MAR1", "MAR2", "MNAR"],
        help="Missing-label mechanism applied to the training labels only.",
    )
    parser.add_argument("--missing-proba", type=float, default=0.3, help="Missing-label probability in train.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the run.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of samples assigned to test.")
    parser.add_argument("--valid-size", type=float, default=0.2, help="Fraction of samples assigned to validation.")
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Threshold for dropping highly correlated features during preprocessing.",
    )
    parser.add_argument(
        "--unlabeled-imputation-type",
        default="logistic",
        choices=["logistic", "knn", "prior"],
        help="Imputation strategy used only for `unlabeled_logreg`.",
    )
    parser.add_argument(
        "--mar1-column-index",
        type=int,
        default=0,
        help="Feature index used by the MAR1 missingness generator.",
    )
    return parser.parse_args()


def main():
    """Run one experiment and print the result as formatted JSON."""
    args = parse_args()

    missingness_kwargs = {}
    if args.missingness == "MAR1":
        missingness_kwargs["missing_influence_col_index"] = args.mar1_column_index

    result = run_single_experiment(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        missingness=args.missingness,
        missing_proba=args.missing_proba,
        random_state=args.random_state,
        test_size=args.test_size,
        valid_size=args.valid_size,
        correlation_threshold=args.correlation_threshold,
        unlabeled_imputation_type=args.unlabeled_imputation_type,
        missingness_kwargs=missingness_kwargs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
