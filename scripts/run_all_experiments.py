"""Run the configured grid of experiments and save the results to CSV."""

import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from project1.experiments.aggregation import aggregate_results, save_results_to_csv
from project1.experiments.configs import (
    DEFAULT_UNLABELED_LABEL_COMPLETION_METHOD,
    EXPERIMENT_DATASETS,
    EXPERIMENT_METHODS,
    EXPERIMENT_MISSING_RATES,
    EXPERIMENT_SEEDS,
    MISSINGNESS_SCHEMES,
)
from project1.experiments.runner import run_single_experiment


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"


def build_experiment_grid():
    """Return the full configured experiment grid as a list of configuration dictionaries."""
    grid = []
    for dataset, scheme, method, seed, missing_rate in product(
        EXPERIMENT_DATASETS,
        MISSINGNESS_SCHEMES,
        EXPERIMENT_METHODS,
        EXPERIMENT_SEEDS,
        EXPERIMENT_MISSING_RATES,
    ):
        config = {
            "dataset": dataset,
            "scheme": scheme,
            "method": method,
            "seed": seed,
            "missing_rate": missing_rate,
        }
        if method == "unlabeled":
            config["label_completion_method"] = DEFAULT_UNLABELED_LABEL_COMPLETION_METHOD
        grid.append(config)
    return grid


def build_error_result(config, exc):
    """Convert a failed configuration into an explicit error result record."""
    error_record = {
        "dataset": config["dataset"],
        "scheme": config["scheme"],
        "method": config["method"],
        "seed": config["seed"],
        "missing_rate": config["missing_rate"],
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
        "status": "error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    if "label_completion_method" in config:
        error_record["label_completion_method"] = config["label_completion_method"]
    return error_record


def run_all_experiments():
    """Run all configured experiments and return the collected list of result records."""
    experiment_grid = build_experiment_grid()
    results = []
    failed_count = 0

    total_experiments = len(experiment_grid)
    for index, config in enumerate(experiment_grid, start=1):
        log_parts = [
            f"[{index}/{total_experiments}]",
            f"dataset={config['dataset']}",
            f"scheme={config['scheme']}",
            f"method={config['method']}",
            f"seed={config['seed']}",
            f"missing_rate={config['missing_rate']}",
        ]
        if config["method"] == "unlabeled":
            log_parts.append(f"label_completion={config['label_completion_method']}")
        print("Running", ", ".join(log_parts))

        try:
            result = run_single_experiment(**config)
        except Exception as exc:
            failed_count += 1
            print(
                "Failed "
                f"[{index}/{total_experiments}], "
                f"dataset={config['dataset']}, "
                f"scheme={config['scheme']}, "
                f"method={config['method']}, "
                f"seed={config['seed']}, "
                f"missing_rate={config['missing_rate']} -> "
                f"{type(exc).__name__}: {exc}"
            )
            results.append(build_error_result(config, exc))
            continue

        result["status"] = "ok"
        results.append(result)

    print(f"Completed full run with {len(results) - failed_count} successful experiments and {failed_count} failures.")
    return results


def build_output_paths():
    """Create timestamped output paths for raw and aggregated experiment tables."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_results_path = OUTPUT_DIR / f"all_experiments_results_{timestamp}.csv"
    aggregated_results_path = OUTPUT_DIR / f"all_experiments_summary_{timestamp}.csv"
    return raw_results_path, aggregated_results_path


def main():
    """Run the full configured experiment grid and save raw plus aggregated CSV outputs."""
    raw_results_path, aggregated_results_path = build_output_paths()

    results = run_all_experiments()
    save_results_to_csv(results, raw_results_path)

    aggregated_df = aggregate_results(results)
    aggregated_results_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated_df.to_csv(aggregated_results_path, index=False)

    print(f"Saved raw results to: {raw_results_path}")
    print(f"Saved aggregated results to: {aggregated_results_path}")


if __name__ == "__main__":
    main()
