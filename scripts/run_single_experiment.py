"""Run a single baseline or unlabeled-logreg experiment from the command line."""

import argparse
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
    parser.add_argument("--dataset", required=True, choices=get_supported_datasets(), help="Dataset used in the run.")
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
    parser.add_argument("--seed", required=True, type=int, help="Random seed for the full run.")
    parser.add_argument(
        "--label-completion-method",
        default="logistic",
        choices=["logistic", "knn", "prior"],
        help="Label-completion method used only when --method unlabeled.",
    )
    return parser.parse_args()


def _format_result(result):
    """Format the experiment result into a readable terminal summary."""
    ordered_keys = [
        "dataset",
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
    lines = ["Single experiment result"]
    for key in ordered_keys:
        if key in result:
            lines.append(f"{key}: {result[key]}")
    return "\n".join(lines)


def main():
    """Run one experiment and print the result in a readable terminal format."""
    args = parse_args()

    result = run_single_experiment(
        dataset=args.dataset,
        scheme=args.scheme,
        method=args.method,
        missing_rate=args.missing_rate,
        seed=args.seed,
        label_completion_method=args.label_completion_method,
    )
    print(_format_result(result))


if __name__ == "__main__":
    main()
